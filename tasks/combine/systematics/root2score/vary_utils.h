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

template <typename T, typename S>
ROOT::RVec<float> evaluate(T cset, const ROOT::RVec<float> &input, const S &name) {
    int size = input.size();
    ROOT::RVec<float> out(size);
    for (int i = 0; i < size; i++) {
        out[i] = cset->evaluate({input[i], name});
    }
    
    return out;
}

template <typename T, typename S>
ROOT::RVec<float> evaluate_btag(const T &cset,
                                const S &name,
                                const ROOT::RVec<int> &had_flav,
                                const ROOT::RVec<float> &abseta,
                                const ROOT::RVec<float> &pt,
                                const ROOT::RVec<float> &btag) {

    int len_ev = had_flav.size();
    ROOT::RVec<float> out(len_ev);
    for (int i = 0; i < len_ev; i++) {
        std::string name_to_use=name;
        if ((had_flav[i] == 4 and (name.find("cferr") == std::string::npos)) || (had_flav[i] == 5 and (name.find("cferr") != std::string::npos))) {
            name_to_use="central";
        }
        out[i]=cset->evaluate({name_to_use, had_flav[i], abseta[i], pt[i], btag[i]});
        //std::vector < correction::Variable::Type> input;
    return out;
    }
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