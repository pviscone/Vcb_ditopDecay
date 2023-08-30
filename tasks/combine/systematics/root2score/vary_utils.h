template <typename T>
ROOT::RVec<float> evaluate(T cset, const std::vector<ROOT::RVec<float>> &inputs) {
    int size = inputs[0].size();
    ROOT::RVec<float> out(size);
    for (int i = 0; i < size; i++) {
        std::vector<correction::Variable::Type> in;
        for (auto const &input : inputs) {
            in.push_back(input[i]);
        }
        out[i] = cset->evaluate(in);
    }
    return out;
}

template <typename T>
ROOT::RVec<float> vary_btag(const T &btag,
                            const std::string &name,
                            const ROOT::RVec<int> &had_flav,
                            const ROOT::RVec<float> &abseta,
                            const ROOT::RVec<float> &pt,
                            const ROOT::RVec<float> &deepJetB,
                            const ROOT::RVec<float> &weight) {
    int len_ev = had_flav.size();
    ROOT::RVec<float> out(len_ev);
    for (int i = 0; i < len_ev; i++) {

        if (abseta[i] >= 2.5) {
            out[i] = 1.;
        } else {
            if ((had_flav[i] == 4 and (name.find("cferr") == std::string::npos)) || ((had_flav[i] == 5 || had_flav[i] == 0) and (name.find("cferr") != std::string::npos))) {
                out[i] = 1.;
            } else {
                out[i] = btag->evaluate({name, had_flav[i], abseta[i], pt[i], deepJetB[i]}) / btag->evaluate({"central", had_flav[i], abseta[i], pt[i], deepJetB[i]});
            }
        }
    }
    return weight*out;
}

template <typename T>
ROOT::RVec<float> vary_ctag(const T &cset,
                            const std::string &name,
                            const ROOT::RVec<int> &had_flav,
                            const ROOT::RVec<float> &CvL,
                            const ROOT::RVec<float> &CvB,
                            const ROOT::RVec<float> &weight) {

    int len_ev = had_flav.size();
    ROOT::RVec<float> out(len_ev);
    for (int i = 0; i < len_ev; i++) {
        out[i] = cset->evaluate({name, had_flav[i], CvL[i], CvB[i]}) / cset->evaluate({"central", had_flav[i], CvL[i], CvB[i]});
    }
    return weight*out;
}

template <typename T, typename S>
ROOT::RVec<float> evaluate(T cset, const ROOT::RVec<float> &input, const S &name) {
    int size = input.size();
    ROOT::RVec<float> out(size);
    for (int i = 0; i < size; i++) {
        out[i] = cset->evaluate({input[i], name});
    }
    
    return out;
}

template <typename T, typename C>
ROOT::RVec<float> evaluate_btag(const T &cset,
                                const std::string &name,
                                const ROOT::RVec<int> &had_flav,
                                const ROOT::RVec<float> &abseta,
                                const ROOT::RVec<float> &pt,
                                const ROOT::RVec<float> &btag,
                                const C &ctag,
                                const ROOT::RVec<float> &CvL,
                                const ROOT::RVec<float> &CvB) {

    int len_ev = had_flav.size();
    ROOT::RVec<float> out(len_ev);
    for (int i = 0; i < len_ev; i++) {

        if (abseta[i]>=2.5){
            out[i]=1.;
        } else{
            if ((had_flav[i] == 4 and (name.find("cferr") == std::string::npos)) || ((had_flav[i] == 5 || had_flav[i] == 0) and (name.find("cferr") != std::string::npos))) {
                out[i] = cset->evaluate({"central", had_flav[i], abseta[i], pt[i], btag[i]})*ctag->evaluate({"central", had_flav[i], CvL[i], CvB[i]});
            } else{
                out[i]=cset->evaluate({name, had_flav[i], abseta[i], pt[i], btag[i]});
            }
        }
        //std::vector < correction::Variable::Type> input;
    }
    return out;
}

template <typename T, typename S>
ROOT::RVec<float> evaluate_ctag(const T &cset,
                                const S &name,
                                const ROOT::RVec<int> &had_flav,
                                const ROOT::RVec<float> &CvL,
                                const ROOT::RVec<float> &CvB
                                ) {

    int len_ev = had_flav.size();
    ROOT::RVec<float> out(len_ev);
    for (int i = 0; i < len_ev; i++) {
        out[i] = cset->evaluate({name, had_flav[i], CvL[i], CvB[i]});
    
    }
    return out;
}

ROOT::RVec<float> TakeIdx(ROOT::RVec<float> const &jetInput, ROOT::RVec<float> const &genInput, ROOT::RVec<int> const &idxs) {
    int size = idxs.size();
    ROOT::RVec<float> out(size);
    for (int i = 0; i < size; i++) {
        if(idxs[i]<0){
            out[i] = jetInput[i];
        }else{
            out[i] = genInput[idxs[i]];
        }
    }
    return out;
}

auto JERC = correction::CorrectionSet::from_file("json/jet_jerc.json");
auto btagging = correction::CorrectionSet::from_file("json/btagging.json");
auto ctagging = correction::CorrectionSet::from_file("json/ctagging.json");