from root2score.rdf2torch.rdf2torch import rdf2torch

def convert2torch(rdf_dict):
    torch_dict={}
    for cut in rdf_dict:
        torch_dict[cut]={}
        for dataset in rdf_dict[cut]:
            print(f"{cut}: {dataset}")
            torch_dict[cut][dataset]={}
            for syst in rdf_dict[cut][dataset]:
                if "signal" in dataset:
                    generator="madgraph"
                else:
                    generator="powheg"
                torch_dict[cut][dataset][syst]=rdf2torch(rdf_dict[cut][dataset][syst],cut=cut,generator=generator)
    return torch_dict