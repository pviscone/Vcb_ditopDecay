void RemoveParameterSets(std::string name){
    TFile *file=new TFile(name.c_str(),"update");
    gDirectory->Delete("ParameterSets;1");
    file->Close();
}
