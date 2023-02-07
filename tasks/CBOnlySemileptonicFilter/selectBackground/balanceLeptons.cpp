using namespace ROOT;

void balanceLeptons(std::string filename,std::string leptonFrom){

    auto original_df=RDataFrame("Events",filename.c_str());
    //Total events*|Vckm|**2/(3*2*sum(|Vckm|**2))
    int NperLepton_cs=62071;
    int NperLepton_cd=3112;
    int NperLepton_ud=60410;
    int NperLepton_us=3211;
    std::string outName=filename+"_new";
    std::string outName_e=outName+"_e";
    std::string outName_mu=outName+"_mu";
    std::string outName_tau=outName+"_tau";

    if(leptonFrom=="leptonFromTbar"){

        auto df=original_df.Filter("!(LHEPart_pdgId[3]==4 && LHEPart_pdgId[4]==-5)");

        auto cs=df.Filter("LHEPart_pdgId[3]==4 && LHEPart_pdgId[4]==-3");
        auto cd=df.Filter("LHEPart_pdgId[3]==4 && LHEPart_pdgId[4]==-1");
        auto ud=df.Filter("LHEPart_pdgId[3]==2 && LHEPart_pdgId[4]==-1");
        auto us=df.Filter("LHEPart_pdgId[3]==2 && LHEPart_pdgId[4]==-3");

        auto e_cs=cs.Filter("LHEPart_pdgId[6]==11").Range(NperLepton_cs);
        auto mu_cs=cs.Filter("LHEPart_pdgId[6]==13").Range(NperLepton_cs);
        auto tau_cs=cs.Filter("LHEPart_pdgId[6]==15").Range(NperLepton_cs);

        auto e_cd=cd.Filter("LHEPart_pdgId[6]==11").Range(NperLepton_cd);
        auto mu_cd=cd.Filter("LHEPart_pdgId[6]==13").Range(NperLepton_cd);
        auto tau_cd=cd.Filter("LHEPart_pdgId[6]==15").Range(NperLepton_cd);


        auto e_ud=ud.Filter("LHEPart_pdgId[6]==11").Range(NperLepton_ud);
        auto mu_ud=ud.Filter("LHEPart_pdgId[6]==13").Range(NperLepton_ud);
        auto tau_ud=ud.Filter("LHEPart_pdgId[6]==15").Range(NperLepton_ud);


        auto e_us=us.Filter("LHEPart_pdgId[6]==11").Range(NperLepton_us);
        auto mu_us=us.Filter("LHEPart_pdgId[6]==13").Range(NperLepton_us);
        auto tau_us=us.Filter("LHEPart_pdgId[6]==15").Range(NperLepton_us);

        e_cs.Snapshot("Events","e_cs_leptFromTbar.root");
        mu_cs.Snapshot("Events","mu_cs_leptFromTbar.root");
        tau_cs.Snapshot("Events","tau_cs_leptFromTbar.root");

        e_cd.Snapshot("Events","e_cd_leptFromTbar.root");
        mu_cd.Snapshot("Events","mu_cd_leptFromTbar.root");
        tau_cd.Snapshot("Events","tau_cd_leptFromTbar.root");

        e_us.Snapshot("Events","e_us_leptFromTbar.root");
        mu_us.Snapshot("Events","mu_us_leptFromTbar.root");
        tau_us.Snapshot("Events","tau_us_leptFromTbar.root");

        e_ud.Snapshot("Events","e_ud_leptFromTbar.root");
        mu_ud.Snapshot("Events","mu_ud_leptFromTbar.root");
        tau_ud.Snapshot("Events","tau_ud_leptFromTbar.root");

    }
    else if(leptonFrom=="leptonFromT"){

        auto df=original_df.Filter("!(LHEPart_pdgId[6]==5 && LHEPart_pdgId[7]==-4)");


        auto cs=df.Filter("LHEPart_pdgId[6]==3 && LHEPart_pdgId[7]==-4");
        auto cd=df.Filter("LHEPart_pdgId[6]==1 && LHEPart_pdgId[7]==-4");
        auto ud=df.Filter("LHEPart_pdgId[6]==1 && LHEPart_pdgId[7]==-2");
        auto us=df.Filter("LHEPart_pdgId[6]==3 && LHEPart_pdgId[7]==-2");


        auto e_cs=cs.Filter("LHEPart_pdgId[3]==-11").Range(NperLepton_cs);
        auto mu_cs=cs.Filter("LHEPart_pdgId[3]==-13").Range(NperLepton_cs);
        auto tau_cs=cs.Filter("LHEPart_pdgId[3]==-15").Range(NperLepton_cs);

        auto e_cd=cd.Filter("LHEPart_pdgId[3]==-11").Range(NperLepton_cd);
        auto mu_cd=cd.Filter("LHEPart_pdgId[3]==-13").Range(NperLepton_cd);
        auto tau_cd=cd.Filter("LHEPart_pdgId[3]==-15").Range(NperLepton_cd);


        auto e_ud=ud.Filter("LHEPart_pdgId[3]==-11").Range(NperLepton_ud);
        auto mu_ud=ud.Filter("LHEPart_pdgId[3]==-13").Range(NperLepton_ud);
        auto tau_ud=ud.Filter("LHEPart_pdgId[3]==-15").Range(NperLepton_ud);


        auto e_us=us.Filter("LHEPart_pdgId[3]==-11").Range(NperLepton_us);
        auto mu_us=us.Filter("LHEPart_pdgId[3]==-13").Range(NperLepton_us);
        auto tau_us=us.Filter("LHEPart_pdgId[3]==-15").Range(NperLepton_us);

        e_cs.Snapshot("Events","e_cs_leptFromT.root");
        mu_cs.Snapshot("Events","mu_cs_leptFromT.root");
        tau_cs.Snapshot("Events","tau_cs_leptFromT.root");

        e_cd.Snapshot("Events","e_cd_leptFromT.root");
        mu_cd.Snapshot("Events","mu_cd_leptFromT.root");
        tau_cd.Snapshot("Events","tau_cd_leptFromT.root");

        e_us.Snapshot("Events","e_us_leptFromT.root");
        mu_us.Snapshot("Events","mu_us_leptFromT.root");
        tau_us.Snapshot("Events","tau_us_leptFromT.root");

        e_ud.Snapshot("Events","e_ud_leptFromT.root");
        mu_ud.Snapshot("Events","mu_ud_leptFromT.root");
        tau_ud.Snapshot("Events","tau_ud_leptFromT.root");



    }
    exit(0);
}
