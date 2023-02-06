using namespace ROOT;

void balanceLeptons(std::string filename,std::string leptonFrom){

    auto original_df=RDataFrame("Events",filename.c_str());
    //Total events/(3*2)
    int NperLepton=128804;
    std::string outName=filename+"_new";
    std::string outName_e=outName+"_e";
    std::string outName_mu=outName+"_mu";
    std::string outName_tau=outName+"_tau";

    if(leptonFrom=="leptonFromTbar"){

        auto df=original_df.Filter("LHEPart_pdgId[3]!=4 && LHEPart_pdgId[4]!=-5");
        auto e=df.Filter("LHEPart_pdgId[6]==11").Range(NperLepton);
        auto mu=df.Filter("LHEPart_pdgId[6]==13").Range(NperLepton);
        auto tau=df.Filter("LHEPart_pdgId[6]==15").Range(NperLepton);

        e.Snapshot("Events",outName_e.c_str());
        mu.Snapshot("Events",outName_mu.c_str());
        tau.Snapshot("Events",outName_tau.c_str());

    }
    else if(leptonFrom=="leptonFromT"){

        auto df=original_df.Filter("LHEPart_pdgId[6]!=5 && LHEPart_pdgId[7]!=-4");
        auto e=df.Filter("LHEPart_pdgId[3]==-11").Range(NperLepton);
        auto mu=df.Filter("LHEPart_pdgId[3]==-13").Range(NperLepton);
        auto tau=df.Filter("LHEPart_pdgId[3]==-15").Range(NperLepton);


        e.Snapshot("Events",outName_e.c_str());
        mu.Snapshot("Events",outName_mu.c_str());
        tau.Snapshot("Events",outName_tau.c_str());

    }
    exit(0);
}
