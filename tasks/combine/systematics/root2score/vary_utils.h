#include "json.hpp"
using json = nlohmann::json;
std::ifstream b("json/btag_normCorrections.json");
std::ifstream c("json/ctag_normCorrections.json");
json j_b = json::parse(b);
json j_c = json::parse(c);

template <typename T>
ROOT::RVec<float> evaluate_btag(const T &cset,
                                const ROOT::RVec<int> &had_flav,
                                const ROOT::RVec<float> &abseta,
                                const ROOT::RVec<float> &pt,
                                const ROOT::RVec<float> &btag) {
    int len_ev = had_flav.size();
    ROOT::RVec<float> out(len_ev);
    for (int i = 0; i < len_ev; i++) {
        std::string flav = std::to_string(had_flav[i]);
        float w_central = j_b[flav]["central"];
        if (abseta[i] >= 2.5) {
            out[i] = 1.;
        } else {
            out[i] = cset->evaluate({"central", had_flav[i], abseta[i], pt[i], btag[i]})/w_central;
        }
    }
    return out;
}

template <typename T>
ROOT::RVec<float> evaluate_ctag(const T &cset,
                                const ROOT::RVec<int> &had_flav,
                                const ROOT::RVec<float> &CvL,
                                const ROOT::RVec<float> &CvB) {

    int len_ev = had_flav.size();
    ROOT::RVec<float> out(len_ev);
    
    for (int i = 0; i < len_ev; i++) {
        std::string flav = std::to_string(had_flav[i]);
        float w_central=j_c[flav]["central"];
        out[i] = cset->evaluate({"central", had_flav[i], CvL[i], CvB[i]})/w_central;
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
        std::string flav = std::to_string(had_flav[i]);
        float w=j_b[flav][name];
        float w_central = j_b[flav]["central"];
        if (abseta[i] >= 2.5) {
            out[i] = 1.;
        } else {
            if ((had_flav[i] == 4 and (name.find("cferr") == std::string::npos)) || ((had_flav[i] == 5 || had_flav[i] == 0) and (name.find("cferr") != std::string::npos))) {
                out[i] = 1.;
            } else {
                out[i] = (btag->evaluate({name, had_flav[i], abseta[i], pt[i], deepJetB[i]})/w) / (btag->evaluate({"central", had_flav[i], abseta[i], pt[i], deepJetB[i]})/w_central);
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
        std::string flav = std::to_string(had_flav[i]);
        float w=j_c[flav][name];
        float w_central=j_c[flav]["central"];
        out[i] = (cset->evaluate({name, had_flav[i], CvL[i], CvB[i]})/w)/(cset->evaluate({"central", had_flav[i], CvL[i], CvB[i]})/w_central);
    }
    return weight*out;
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

auto JERC = correction::CorrectionSet::from_file("json/jet_jerc.json");
auto btagging = correction::CorrectionSet::from_file("json/btagging.json");
auto ctagging = correction::CorrectionSet::from_file("json/ctagging.json");