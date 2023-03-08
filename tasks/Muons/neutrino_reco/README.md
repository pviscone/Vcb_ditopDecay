> ## TODO
> 
> - Reconstruct the $p_z$ of the neutrino
> 
> - Plot $R_{min}$ between the leptonic b jet ( $t \to b (W\to l \nu)$ )  and the muon
> 
> - Reconstruct the top mass (selecting the jet with $R_{min}<0.4$)
> 
> - Reconstruct the top mass using the other jets
> 
> - Do the same plots with the btagDeepJetB

## Neutrino reconstrucion

Considering the lepton massless

$$
p_W^2=(p_l+p_\nu)^2 \\
\\ \;
\\ \;
\implies 4p_{l,t} p_{\nu,z}^2-4p_{l,z}(M_W^2+2 \vec{p_{l,t}} \cdot \vec{p_{\nu,t}})p_{\nu,z}-\\
\\
-(M_W^2+2 \vec{p_{l,t}} \cdot \vec{p_{\nu,t}})^2+4E_l^2p_{\nu,t}^2=0
$$

Solve for $p_{\nu,z}$

If $\Delta>0$ choose the smallest solution

If $\Delta<0$ impose it to 0 (take only the real part of the solution)

This approach is not based on physical reasons.

There are other approachs (also using flows and NN) but here was used the simplest method

> Papers on the topic:
> 
> - [ν-Flows: conditional neutrino regression](https://arxiv.org/abs/2207.00664))
> 
> - [Comparing Traditional and Deep-Learning Techniques of Kinematic Reconstruction for polarisation Discrimination in Vector Boson Scattering](https://arxiv.org/abs/2008.05316)
> 
> - [Evidence for the charge asymmetry in $pp \rightarrow t\bar{t}$ production at $\sqrt{s}= 13$ TeV with the ATLAS detector](https://arxiv.org/abs/2208.12095)

## Observation

- There is a strange modulation in MET_phi (It should be uniform)
