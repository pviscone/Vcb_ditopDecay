void RemoveEvents(std::string name){
    TFile *file=new TFile(name.c_str(),"update");
    gDirectory->Delete("Events;1");
    file->Close();

    exit(0);
}
