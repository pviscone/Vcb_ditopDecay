template <typename T>
ROOT::RVec<float> evaluate(T cset, std::vector<ROOT::RVec<float>> const &inputs) {
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
ROOT::RVec<float> evaluate(T cset, ROOT::RVec<float> const &input, const S &name) {
    int size = input.size();
    ROOT::RVec<float> out(size);
    for (int i = 0; i < size; i++) {
        out[i] = cset->evaluate({input[i], name});
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