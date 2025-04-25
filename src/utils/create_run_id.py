from datetime import datetime
def create_run_id(codetype, rand_seed):
    return datetime.now().strftime("%y%m%d_%H%M%S") + "_type_" + codetype.lower() + "_seed_" + str(rand_seed)