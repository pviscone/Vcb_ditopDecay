//global pdg database
TDatabasePDG* pdgDatabase=new TDatabasePDG();


/**
 * @brief Function that, given the name of the particle, returns the PDG code
 * If the particle is not found, the function returns 0 (the Rootino)
 *
 * @param name particle name
 * @return int pdgId
 */
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


/**
 * @brief Function that, given the PDG code, returns the name of the particle
 * If the particle does not exist, it returns "Rootino"
 *
 * @param id pdgId
 * @return std::string particle name
 */
 */
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
