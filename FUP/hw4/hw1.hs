
import SedmaDatatypes

data Players =  A | B | C | D deriving (Show, Eq)

--The first round is led by player A.
--test OK
clockWiseOrder :: Players -> Players
clockWiseOrder A = B
clockWiseOrder B = C
clockWiseOrder C = D
clockWiseOrder D = A

--test OK
cardValue :: Card -> Int
--if cardValue (Card _ RA) then 10
cardValue (Card _ RA) = 10
cardValue (Card _ R10) = 10
cardValue (Card _ R7) = 0
cardValue (Card _ R8) = 0
cardValue (Card _ R9) = 0
cardValue (Card _ RJ) = 0
cardValue (Card _ RQ) = 0
cardValue (Card _ RK) = 0

--test OK
teams :: Players -> Team
teams A = AC
teams B = BD
teams C = AC
teams D = BD

--test OK
rankEq :: Card -> Card -> Bool
rankEq (Card _ RA) (Card _ RA) = True
rankEq (Card _ R10) (Card _ R10) = True
rankEq (Card _ R9) (Card _ R9) = True
rankEq (Card _ R8) (Card _ R8) = True
rankEq (Card _ RK) (Card _ RK) = True
rankEq (Card _ RQ) (Card _ RQ) = True
rankEq (Card _ RJ) (Card _ RJ) = True
rankEq (Card _ R7) _ = True
rankEq _ _ = False


roundsScore :: Cards -> Int
roundsScore (x:ns) = cardValue x + roundsScore ns
roundsScore [] = 0

--There are additional 10 points for the winner of the last trick. 
--(score of A + score of C (+10 if A or C), score of B + score of D(+10 if D or B))
--test OK
allScore :: [Cards] -> Maybe Players -> (Int,Int)
allScore [scoreA, scoreB, scoreC, scoreD] (Just A) = (10 + roundsScore scoreA + roundsScore scoreC, roundsScore scoreB + roundsScore scoreD)
allScore [scoreA, scoreB, scoreC, scoreD] (Just B) = (roundsScore scoreA + roundsScore scoreC,10 + roundsScore scoreB + roundsScore scoreD)
allScore [scoreA, scoreB, scoreC, scoreD] (Just C) = (10 + roundsScore scoreA + roundsScore scoreC, roundsScore scoreB + roundsScore scoreD)
allScore [scoreA, scoreB, scoreC, scoreD] (Just D) = (roundsScore scoreA + roundsScore scoreC,10 + roundsScore scoreB + roundsScore scoreD)
allScore [scoreA, scoreB, scoreC, scoreD] Nothing = (roundsScore scoreA + roundsScore scoreC,roundsScore scoreB + roundsScore scoreD)
allScore _ _ = (0,0)

--if one of team was not able to win a single trick 
noSingleTrick :: Team -> [Cards] -> Bool
noSingleTrick AC [scoreA, scoreB, scoreC, scoreD] = null scoreA && null scoreC
noSingleTrick BD [scoreA, scoreB, scoreC, scoreD] = null scoreB && null scoreD

--test OK
forNext :: Maybe Players -> Players
forNext Nothing = error "IDK.fromJust: None"
forNext (Just x) = x

--test OK
ifWinner :: Card -> Cards -> Players -> Maybe Players
ifWinner first stock player = if null stock then Nothing else if
    nxWin == Nothing then if rankEq (head stock) first then Just player else Nothing else
        nxWin
        where
            nxWin = (ifWinner first (tail stock) (clockWiseOrder player))


--test OK
nextLeader :: Cards -> Players -> Maybe Players
nextLeader [] _ = Nothing
nextLeader stock player = 
    if nxWin == Nothing then winner else nxWin
    where
        winner = ifWinner (head stock) (take 4 stock) player
        nxWin = nextLeader (drop 4 stock) (if winner == Nothing then player else (forNext winner))


try :: Cards -> Players -> [Cards]
try [] _ = [[], [], [], []]
try fromStock player
    |player == A = zipWith (++) [take 4 fromStock, [], [], []] have
    |player == C = zipWith (++) [[], [], take 4 fromStock, []] have
    |player == B = zipWith (++) [[], take 4 fromStock, [], []] have
    |player == D = zipWith (++) [[], [], [], take 4 fromStock] have
    |winner == A = zipWith (++) [take 4 fromStock, [], [], []] have
    |winner == C = zipWith (++) [[], [], take 4 fromStock, []] have
    |winner == B = zipWith (++) [[], take 4 fromStock, [], []] have
    |otherwise = zipWith (++) [[], [], [], take 4 fromStock] have
    where
        Just winner = ifWinner (head fromStock) (take 4 fromStock) player
        have = try (drop 4 fromStock) winner 
    
replay :: Cards -> Maybe Winner
    
replay [] = Nothing
replay stock 
    |length (stock) /= 32 = Nothing
    |scoreAC > scoreBD && noSingleTrick BD tryCard = Just (AC, Three)
    |scoreAC < scoreBD && noSingleTrick AC tryCard = Just (BD, Three)
    |scoreAC > scoreBD && scoreBD == 0 = Just (AC, Two)
    |scoreAC < scoreBD && scoreAC == 0 = Just (BD, Two)
    |scoreAC > scoreBD && scoreBD /= 0 = Just (AC, One)
    |scoreAC < scoreBD && scoreAC /= 0 = Just (BD, One)
    |otherwise = Nothing
    where
        tryCard = try stock A
        winLast = nextLeader stock A
        (scoreAC, scoreBD) = (allScore tryCard winLast)
