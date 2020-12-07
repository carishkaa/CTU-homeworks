from csp import Constraint, CSP
from typing import Dict, List, Optional, Any

import json, sys, argparse, numpy as np


# --------------------------------------------------------------------------
class TaxisConstraint(Constraint):
    def __init__(self, variables: List, customers: Dict, taxis: Dict) -> None:
        super().__init__(variables)
        self.customers = customers
        self.taxis = taxis

    def satisfied(self, assignment: Dict) -> bool:
        taxis_occupancy = {}
        for cid, tid in assignment.items():
            customer = self.customers[cid]
            taxi = self.taxis[tid]

            # empty taxi
            if tid not in taxis_occupancy and taxi["capacity"] > 0:
                taxis_occupancy[tid] = [cid]
                continue

            # customers in the taxi
            cur_occupancy = taxis_occupancy[tid]

            # check if taxi is not full
            if len(cur_occupancy) >= taxi["capacity"]:
                return False

            # check if they all have same origin and destination points
            for cid2 in cur_occupancy:
                if self.customers[cid2]["origin"] != customer["origin"]:
                    return False
                if self.customers[cid2]["destination"] != customer["destination"]:
                    return False

            cur_occupancy.append(cid)
            taxis_occupancy[tid] = cur_occupancy
        return True


# --------------------------------------------------------------------------
def load_input(file):
    if file == '-':
        data = json.load(sys.stdin)
    else:
        with open(file, 'r') as fp:
            data = json.load(fp)
    return data


# --------------------------------------------------------------------------
if __name__ == "__main__":
    # setup arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help="input_file, use '-' for standard input")
    args = parser.parse_args()

    # load data
    data = load_input(args.input_file)
    # print(json.dumps(data, indent=4, separators=(',', ': ')))

    customers = {}
    taxis = {}
    for customer in data["customers"]:
        customers[customer["id"]] = customer
    for taxi in data["taxis"]:
        taxis[taxi["id"]] = taxi

    # prepare variables and their domains
    variables: List = []
    domains: Dict = {}

    for customer in data["customers"]:
        taxi_ids = []
        for taxi in data["taxis"]:
            if customer["class"] < taxi["class"]:
                continue
            if customer["origin"] in taxi["serves"] and customer["destination"] in taxi["serves"]:
                taxi_ids.append(taxi["id"])
        # sort by capacity
        taxi_capacity = [taxis[tid]["capacity"] for tid in taxi_ids]
        indices = np.array(taxi_capacity).argsort()[::-1]
        domain = [taxi_ids[i] for i in indices]

        domains[customer["id"]] = domain

    # MCV (sort variables by domains length)
    domains_values = list(domains.values())
    domains_keys = list(domains.keys())
    domains_count = [len(domains_values[i]) for i in range(len(domains))]
    indices = np.argsort(domains_count)
    variables = [domains_keys[i] for i in indices]

    # create CSP
    csp: CSP = CSP(variables, domains)
    csp.add_constraint(TaxisConstraint(variables, customers, taxis))

    # find solution
    solution: Optional[Dict] = csp.backtracking_search()

    # print it out
    if solution is None:
        print("{}")
    else:
        print(json.dumps(solution))
