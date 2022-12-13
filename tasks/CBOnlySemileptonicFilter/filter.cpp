using namespace ROOT;

bool selectCB(const RVec<float> &vec){
    if(vec[3]==4 && vec[4]==-5){
        return true;
    } else {return false;}
}


void filter(){
    EnableImplicitMT();
    RDataFrame df("Events", "../ttbarKinematics/madgraph/73B85577-0234-814E-947E-7DCFC1275886.root");

    auto cb = df.Filter("selectCB(LHEPart_pdgId)");
    cb.Snapshot("Events","./cb.root");

    auto others = df.Filter("!selectCB(LHEPart_pdgId)");
    others.Snapshot("Events","./others.root");
}