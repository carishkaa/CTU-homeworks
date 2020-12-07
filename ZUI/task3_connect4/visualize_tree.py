from testing_games import TreeGame, create_tree_game_score
from PyQt5 import QtGui  # type: ignore
from PyQt5.QtGui import QPen, QBrush, QPainter  # type: ignore
from PyQt5.QtCore import Qt, QRectF, QTimer  # type: ignore
from PyQt5.QtWidgets import QApplication, QMainWindow  # type: ignore
from agent import DeterministicAgent
from collections import deque
import sys


class Node:
    def __init__(self, state, true_value=None):
        self.state = state
        self.children = []
        self.value = None
        self.true_value = true_value
        self.visited = 0
        self.pruned = False


class Tree:
    def __init__(self, root, depth, width):
        self.root = root
        self.root.visited = 1
        self.width = width
        self.depth = depth


class TreeEmulator:
    def __init__(self, values, branching_factor=3, depth=3):
        assert len(values) == branching_factor ** depth
        self.game = TreeGame(depth + 1, branching_factor, None)
        self.heuristic_fn = create_tree_game_score(depth, values)
        self.branching_factor = branching_factor
        self.depth = depth
        self._finished = True
        self._tree, treeMap = self.createTree(self.game.clone())
        self._updates = deque(self.run_collect_updates(treeMap))

    def next(self):
        self._updates.popleft()()
        return len(self._updates)

    def run_collect_updates(self, treeMap):
        game = self.game.clone()
        updates = []
        openNodes = set()

        def increment(node):
            node.visited += 1

        def set_value(node, value):
            node.value = value

        def set_pruned(node):
            node.pruned = True

        # Override alphabeta
        heuristic_fn = self.heuristic_fn
        depth = self.depth

        class Agent(DeterministicAgent):
            def __init__(self):
                super().__init__(heuristic_fn, 0, depth)

            def alphabeta(self, game, alpha, beta, depth, color):
                node = treeMap[game._number]
                updates.append(lambda: increment(node))
                if node in openNodes:
                    openNodes.remove(node)
                for child in node.children:
                    openNodes.add(child)

                val = super().alphabeta(game, alpha, beta, depth, color)

                def _set_pruned(child):
                    updates.append(lambda: set_pruned(child))

                for child in node.children:
                    if child in openNodes:
                        _set_pruned(child)
                updates.append(lambda: set_value(node, val))
                return val

        agent = Agent()
        agent.compute_estimate(game)
        return updates

    def createTree(self, game):
        nodeMap = {}

        def getNode(game, depth=self.depth):
            if depth == 0:
                n = Node(game._number, self.heuristic_fn(game, 0))
                nodeMap[n.state] = n
                return n

            n = Node(game._number)
            nodeMap[n.state] = n
            for a in game.legal_actions():
                g = game.clone()
                g.apply(a)
                n.children.append(getNode(g, depth - 1))
            return n

        return Tree(getNode(game), self.depth, self.branching_factor ** self.depth), nodeMap

    @property
    def finished(self):
        return self._finished

    @property
    def tree(self):
        return self._tree


class Window(QMainWindow):
    def __init__(self, emulator, drawFinal=False):
        super().__init__()
        self.top = 150
        self.left = 150
        self.nodeOuterWidth = 45
        self.nodeOuterHeight = 55
        self.nodeSize = 40
        self.globalYOffset = 30
        self.width = (emulator.tree.width + 1) * self.nodeOuterWidth + 100 + self.nodeSize
        self.height = (emulator.tree.depth + 1) * self.nodeOuterHeight + self.globalYOffset * 2 + self.nodeSize
        self.setWindowTitle("Minimax tree visualizer")
        self.setGeometry(self.top, self.left, self.width, self.height)
        self.emulator = emulator
        if drawFinal:
            while self.emulator.next():
                pass
            self.show()
        else:
            self.show()
            self.timer = QTimer()
            self.timer.timeout.connect(self.nextUpdate)
            self.timer.start(1000)

    def nextUpdate(self):
        if not self.emulator.next():
            self.timer.stop()
        self.update()

    def drawTree(self, tree, painter):
        nodeSize = self.nodeSize

        def drawLayer(nodes, layer):
            spacing = self.nodeOuterWidth * self.emulator.branching_factor ** (tree.depth - layer)
            xoffset = self.width / 2 - (len(nodes) / 2 - 0.5) * spacing
            yoffset = layer * self.nodeOuterHeight + self.nodeSize / 2 + self.globalYOffset
            childspacing = self.nodeOuterWidth * self.emulator.branching_factor ** (tree.depth - layer - 1)
            childxoffset = self.width / 2 - (len(nodes) * self.emulator.branching_factor / 2 - 0.5) * childspacing
            for i, n in enumerate(nodes):
                x, y = xoffset + i * spacing, yoffset
                if layer < self.emulator.depth:
                    # draw connecting lines
                    for j, child in enumerate(n.children):
                        chx = childxoffset + (i * self.emulator.branching_factor + j) * childspacing
                        painter.setPen(QPen(Qt.black if not child.pruned else Qt.red,
                                            1 if (not child.visited and not child.pruned) else 3,
                                            Qt.SolidLine if not child.pruned else Qt.DashLine))
                        painter.drawLine(x, y, chx,
                                         (layer + 1) * self.nodeOuterHeight + self.nodeSize / 2 + self.globalYOffset)

                rect = QRectF(x - self.nodeSize / 2, y - self.nodeSize / 2, self.nodeSize, self.nodeSize)
                painter.setPen(QPen(Qt.black, 1 if (not n.visited or n.pruned) else 3, Qt.SolidLine))
                painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
                painter.drawEllipse(rect)
                if n.value is not None or n.true_value is not None:
                    font = painter.font()
                    font.setBold(n.value is not None)
                    painter.setFont(font)
                    painter.drawText(rect, Qt.AlignCenter, str(n.value) if n.value is not None else str(n.true_value))

        nodes = [tree.root]
        layer = 0
        while len(nodes) > 0:
            drawLayer(nodes, layer)
            nodes = [n for p in nodes for n in p.children]
            layer += 1

    def paintEvent(self, event):
        painter = QPainter(self)
        self.drawTree(self.emulator.tree, painter)


if __name__ == "__main__":
    #
    # Tool for visualizing your alphabeta implementation
    # Values can be copied from http://inst.eecs.berkeley.edu/~cs61b/fa14/ta-materials/apps/ab_tree_practice/
    #
    values = [-11, -10, 17, 12, -8, 14, 16, -2, -2, -1, 14, -17, 0, -5, 0, 8, 3, 10, -9, 8, 10, -5, -13, -15, 12, -11,
              -11]

    App = QApplication(sys.argv)
    emulator = TreeEmulator(values)

    window = Window(emulator)
    sys.exit(App.exec())
