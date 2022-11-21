TDatabasePDG* pdgDatabase=new TDatabasePDG();

int pdg(const char * name){
    int id;
    if (id==0){
        std::cerr << "WARNING: there is a ROOTINO" << std::endl;
    }
    try{
        id=pdgDatabase->GetParticle(name)->PdgCode();
    } catch (std::exception &err){
        std::cerr << "The particle name: " << name << " does not exist" << std::endl;
        id=0;

    }
    return id;
}

std::string pdg(int id){
    std::string name;
    try{
        name=pdgDatabase->GetParticle(id)->GetName();
    } catch (std::exception &err){
        std::cerr << "The pdgId: "<< id << " does not exist" <<std::endl;
        name="Rootino";
    }
    return name;
}



void main0(){
    TFile* f=new TFile("ttbar.root");
    TTree* t=(TTree*)f->Get("Events");

    t->SetBranchStatus("*",0);
    t->SetBranchStatus("LHE*",1);

    int N=t->GetEntries();
    for(int i=0;i<N;i++){
        t->GetEntry(i);

    }

}
