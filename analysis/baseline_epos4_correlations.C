// Rishabh Gupta @ 2026
// Baseline EPOS4 p-Lambda and pp-Lambda correlation analysis.
// This macro performs same-event/mixed-event study on EPOS4 data.
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TVector3.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLorentzVector.h"
#include "TSystem.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <iostream>
#include <string>
#include <vector>

struct ParticleRecord {
    TLorentzVector p4;   // momentum four-vector
    TLorentzVector x4;   // position four-vector, it will be later stored as x4.setXZYT(x,y,z,t). Which we will use for source seperation clacualtion |vec(x1) - vec(x2)| and pair restframe seperation r*
}; 

// since we are studying the correlation between them in two levels and also trying to study, can we use this info for the model regularisation in some way.
struct SelectedEvent {  
    std::vector<ParticleRecord> protons;
    std::vector<ParticleRecord> antiProtons;
    std::vector<ParticleRecord> lambdas;
    std::vector<ParticleRecord> antiLambdas;
    int nch = 0; 
};

// defining funcitons to calcualte the variables
double calcKStar(const TLorentzVector& a, const TLorentzVector& b) {
    const double s = (a+b).M2(); 
    if (s <= 0.0 || !std::isfinite(s)) {
        return -1.0;
    }
    const double m1 = a.M(); // invariant mass
    const double m2 = b.M();
    const double term1 = s-(m1+m2)*(m1+m2); 
    const double term2 = s-(m1-m2)*(m1-m2);
    double lambda = term1*term2; // triangle fucntion (Byckling & Kajantie)

    if (lambda < 0.0 && lambda > -1e-8) {
        lambda = 0.0;
    }
    if (lambda < 0.0 || !std::isfinite(lambda)) {
        return -1.0;
    }
    return std::sqrt(lambda)/(2.0*std::sqrt(s));
}

// invariant realtive momentum
double qInv(const TLorentzVector& a, const TLorentzVector& b) {

    TLorentzVector diff = a - b;
    
    double q2 = -diff.M2();
    //unphysical   
    if (q2 < 0.0 && q2 > -1e-8) {
        q2 = 0.0;
    }
    if (q2 < 0.0 || !std::isfinite(q2)) {
        return -1.0;
    }
    return std::sqrt(q2);
}

//three-particle relative momentum
double Q3(const TLorentzVector& a, const TLorentzVector& b, const TLorentzVector& c) {

    const double q12 = qInv(a, b);
    const double q23 = qInv(b, c);
    const double q31 = qInv(c, a);

    if (q12 < 0.0 || q23 < 0.0 || q31 < 0.0) {
        return -1.0;
    }

    return std::sqrt(q12 * q12 + q23 * q23 + q31 * q31);
}

// Source seperation (r)
double sourceSeparationLab(const ParticleRecord& a, const ParticleRecord& b) {
    const double dx = a.x4.X() - b.x4.X();
    const double dy = a.x4.Y() - b.x4.Y();
    const double dz = a.x4.Z() - b.x4.Z();

    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// r* for pairs
double sourceSeparationPairRestFrame(const ParticleRecord& a, const ParticleRecord& b) {
    TLorentzVector pair = a.p4 + b.p4;
    TVector3 beta = pair.BoostVector();

    TLorentzVector dr;
    dr.SetXYZT(a.x4.X() - b.x4.X(), a.x4.Y() - b.x4.Y(), a.x4.Z() - b.x4.Z(), a.x4.T() - b.x4.T()); // setting the four vector as the difference of the two particles
    dr.Boost(-beta); // this transforms the frame

    return dr.Vect().Mag(); // | del(vec(r*)) |
}

// a toy variable to estimate the three body spatial size, taking the net vector in all three directions for now, in future it may include some transformation
double tripletHyperradiusLab(const ParticleRecord& a, const ParticleRecord& b, const ParticleRecord& c) {
    const double rab = sourceSeparationLab(a, b);
    const double rbc = sourceSeparationLab(b, c);
    const double rca = sourceSeparationLab(c, a);

    return std::sqrt(rab*rab + rbc*rbc + rca*rca);
}

bool isChargedEPOS(int id) {
    const int aid = std::abs(id);
    // charged particle used are:  pi,   k ,   p,  sig+, Sig-, Xi-,  Omega-
    const int chargedEPOSIds[] = { 120, 130, 1120, 1130, 2230, 2330, 3331 };

    const int nChargedIds = sizeof(chargedEPOSIds) / sizeof(chargedEPOSIds[0]);

    for (int i = 0; i < nChargedIds; ++i) {
        if (aid == chargedEPOSIds[i]) {
            return true;
        }
    }

    return false;
}

// the correlation function
void CorrelationC(TH1D* same, TH1D* mixed, TH1D* corr, double normLow, double normHigh) {
    if (!same || !mixed || !corr) {
        return;
    }
    const int binLow = same->FindBin(normLow);
    const int binHigh = same->FindBin(normHigh);

    const double sameIntegral = same->Integral(binLow, binHigh);
    const double mixedIntegral = mixed->Integral(binLow, binHigh);

    std::cout<<"\nNormalizing "<< corr->GetName()<<"\n";
    std::cout<<"Same integral  ["<< normLow<< ", "<<normHigh<<"] = "<<sameIntegral<<"\n";
    std::cout<<"Mixed integral ["<<normLow<<", "<<normHigh<<"] = "<<mixedIntegral<<"\n";

    if (sameIntegral > 0.0 && mixedIntegral > 0.0) {
        mixed->Scale(sameIntegral / mixedIntegral);  // C = A/B' where B' is scaled
    } else {
        std::cout<<"WARNING: normalization skipped because one integral is zero.\n";
    }

    corr->Reset();
    corr->Divide(same, mixed, 1.0, 1.0);
}

void savePlot(TH1D* h, const char* outName) {
    if (!h) {
        return;
    }
    
    TCanvas* c = new TCanvas(Form("c_%s", h->GetName()), h->GetTitle(), 900, 700);
    
    h->SetLineWidth(2);
    h->Draw("E");
    c->SaveAs(outName);

    delete c;
}

// using COLZ for plotting
void savePlot2D(TH2D* h, const char* outName) {
    if (!h) { return; }

    TCanvas* c = new TCanvas(Form("c_%s", h->GetName()), h->GetTitle(), 900, 700);
    h->Draw("COLZ");
    c->SaveAs(outName);
    delete c;
}

// this funciton will be used by root, setting default file with 1 million events
void make_epos4_correlations(const char* inputFile = "pp13.6TeVEPOS4-0.root", Long64_t maxEvents = 1000000, double etaMax = 0.8, int mixDepth = 5, int minNch = 0, const char* outputFile = "epos4_pLambda_correlations.root") {
    std::cout<<"\nEPOS4 p-Lambda source and correlation\n";
    std::cout<<"Input file  : "<<inputFile<<"\n";
    std::cout<<"Output file : "<<outputFile<<"\n";
    std::cout<<"Max events  : "<<maxEvents<<"\n";
    std::cout<<"Eta cut     : |eta| < "<<etaMax<<"\n";
    std::cout<<"Mix depth   : "<<mixDepth<<"\n";
    std::cout<<"Min Nch     : "<<minNch<<"\n\n";

    TFile* f = TFile::Open(inputFile);
    if (!f || f->IsZombie()) {
        std::cerr<<"ERROR: Could not open input file: "<<inputFile<<"\n";
        return;
    }
    TTree* t = (TTree*) f->Get("teposevent0"); // tree name used in the default file
    if (!t) {
        std::cerr<<"ERROR: Could not find TTree named teposevent0\n";
        f->ls();
        return;
    }

    constexpr int MAXP = 200000; // max particles per event
 
    Int_t np = 0;               // number of particles per event
    Int_t id[MAXP];             // particle id
    Int_t ist[MAXP];            // flag
    // momentum four-vector
    Float_t px[MAXP];
    Float_t py[MAXP];
    Float_t pz[MAXP];
    Float_t massBranch[MAXP]; 
    // position four-vector
    Float_t xpos[MAXP];
    Float_t ypos[MAXP];
    Float_t zpos[MAXP];
    Float_t tpos[MAXP];

    t->SetBranchStatus("*", 0);    // since the file is big we will consider only those branches which are needed for our analysis, turning off the rest
    t->SetBranchStatus("np", 1);
    t->SetBranchStatus("id", 1);
    t->SetBranchStatus("ist", 1);
    t->SetBranchStatus("px", 1);
    t->SetBranchStatus("py", 1);
    t->SetBranchStatus("pz", 1);
    t->SetBranchStatus("e", 1);
    t->SetBranchStatus("x", 1);
    t->SetBranchStatus("y", 1);
    t->SetBranchStatus("z", 1);
    t->SetBranchStatus("t", 1);
    t->SetBranchAddress("np", &np);
    t->SetBranchAddress("id", id);
    t->SetBranchAddress("ist", ist);
    t->SetBranchAddress("px", px);
    t->SetBranchAddress("py", py);
    t->SetBranchAddress("pz", pz);
    t->SetBranchAddress("e", massBranch);   // in the file 'e' is used as mass-like value
    t->SetBranchAddress("x", xpos);
    t->SetBranchAddress("y", ypos);
    t->SetBranchAddress("z", zpos);
    t->SetBranchAddress("t", tpos);

    // Nch histogram
    TH1D* hNch = new TH1D("hNch", Form("Charged multiplicity; N_{ch}(|#eta|<%.1f); Events", etaMax), 200, 0, 200);
    // proton pT histogram
    TH1D* hProtonPt = new TH1D("hProtonPt", "Selected p/#bar{p}; p_{T} [GeV/c]; Counts", 100, 0.0, 5.0);
    // lambda pT histogram
    TH1D* hLambdaPt = new TH1D("hLambdaPt", "Selected #Lambda/#bar{#Lambda}; p_{T} [GeV/c]; Counts", 100, 0.0, 5.0);
    // same-event p and lambda stored as a function of k* 
    TH1D* hSame_pLambda_kstar = new TH1D("hSame_pLambda_kstar", "Same-event p#Lambda #oplus #bar{p}#bar{#Lambda}; k* [GeV/c]; Counts", 100, 0.0, 5.0);
    // mixed-event p and lambda stored as a function of k*
    TH1D* hMixed_pLambda_kstar = new TH1D("hMixed_pLambda_kstar", "Mixed-event p#Lambda #oplus #bar{p}#bar{#Lambda}; k* [GeV/c]; Counts", 100, 0.0, 5.0);
    // toy correlation function 
    TH1D* hCorr_pLambda_kstar = new TH1D("hCorr_pLambda_kstar", "C_{p#Lambda}(k*); k* [GeV/c]; C(k*) (Toy)", 100, 0.0, 5.0);
    // same-event triplets
    TH1D* hSame_ppLambda_Q3 = new TH1D("hSame_ppLambda_Q3", "Same-event pp#Lambda #oplus #bar{p}#bar{p}#bar{#Lambda}; Q_{3} [GeV/c]; Counts", 80, 0.0, 2.4);
    // mixed-event triplets
    TH1D* hMixed_ppLambda_Q3 = new TH1D("hMixed_ppLambda_Q3", "Mixed-event pp#Lambda #oplus #bar{p}#bar{p}#bar{#Lambda}; Q_{3} [GeV/c]; Counts", 80, 0.0, 2.4);
    // toy triplet correlation
    TH1D* hCorr_ppLambda_Q3 = new TH1D("hCorr_ppLambda_Q3", "C_{pp#Lambda}(Q_{3}); Q_{3} [GeV/c]; C(Q_{3}) (Toy)", 80, 0.0, 2.4);
    // r_lab
    TH1D* hSource_pLambda_rLab = new TH1D("hSource_pLambda_rLab", "Same-event p#Lambda source separation; r_{lab} [fm]; Counts", 120, 0.0, 30.0);
    // r*
    TH1D* hSource_pLambda_rStar = new TH1D("hSource_pLambda_rStar", "Same-event p#Lambda source separation; r* [fm]; Counts", 120, 0.0, 30.0);
    // the net of r in all three directions
    TH1D* hSource_ppLambda_rhoLab = new TH1D("hSource_ppLambda_rhoLab", "Same-event pp#Lambda source hyperradius; #rho_{lab} [fm]; Counts", 120, 0.0, 60.0);
    // 2D map of the relative momentum and radius (k*, r*)
    TH2D* hKStarVsRStar_pLambda = new TH2D("hKStarVsRStar_pLambda", "Same-event p#Lambda source-momentum map; k* [GeV/c]; r* [fm]", 80, 0.0, 5.0, 80, 0.0, 30.0);
    
    // weighted sum
    hNch->Sumw2();
    hProtonPt->Sumw2();
    hLambdaPt->Sumw2();
    hSame_pLambda_kstar->Sumw2();
    hMixed_pLambda_kstar->Sumw2();
    hCorr_pLambda_kstar->Sumw2();
    hSame_ppLambda_Q3->Sumw2();
    hMixed_ppLambda_Q3->Sumw2();
    hCorr_ppLambda_Q3->Sumw2();
    hSource_pLambda_rLab->Sumw2();
    hSource_pLambda_rStar->Sumw2();
    hSource_ppLambda_rhoLab->Sumw2();


    Long64_t nEntries = t->GetEntries();

    if (maxEvents <= 0 || maxEvents > nEntries) {
        maxEvents = nEntries;
    }
    // using double queue vector for mixing buffer
    std::deque<SelectedEvent> mixingBuffer;

    Long64_t acceptedEvents = 0;                  // number of passed events
    // counts selected events
    Long64_t totalSelectedProtons = 0;         
    Long64_t totalSelectedAntiProtons = 0;    
    Long64_t totalSelectedLambdas = 0;
    Long64_t totalSelectedAntiLambdas = 0;
    // pair counter and triplet counters
    Long64_t samePairs = 0;
    Long64_t mixedPairs = 0;
    Long64_t sameTriplets = 0;
    Long64_t mixedTriplets = 0;
    // varibles used for normalisation later
    Long64_t eventsWithP = 0;
    Long64_t eventsWithLambda = 0;
    Long64_t eventsWithBothPLambda = 0;
    Long64_t eventsWithPPLambda = 0;

    // Event loop
    for (Long64_t iev = 0; iev < maxEvents; ++iev) {
        t->GetEntry(iev);
        if (np >= MAXP) {
            std::cerr<<"WARNING: event "<<iev<<" has np = "<<np<<" larger than MAXP = "<<MAXP<<". Skipping event.\n";
            continue;
        }
        SelectedEvent ev;
        // particle loop
        for (int i = 0; i < np; ++i) {
            // EPOS4 active/final particles.
            if (ist[i] != 0) {   // avoid double counting
                continue;
            }
            const int pid = id[i];

            const double m = massBranch[i];

            if (m <= 0.0 || !std::isfinite(m)) {
                continue;
            }

            const double p2 = px[i]*px[i] + py[i]*py[i] + pz[i]*pz[i];
            const double energy = std::sqrt(p2 + m*m);

            TLorentzVector p4;
            p4.SetPxPyPzE(px[i], py[i], pz[i], energy);

            const double eta = p4.Eta();
            // for safety
            if (!std::isfinite(eta)) { continue; }
            // eta cut
            if (std::abs(eta) > etaMax) { continue;}
            // Nch counter
            if (isChargedEPOS(pid)) { ev.nch++; }

            ParticleRecord rec;
            rec.p4 = p4;
            rec.x4.SetXYZT(xpos[i], ypos[i], zpos[i], tpos[i]);
            // updating each particle vector
            if (pid == 1120) {
                ev.protons.push_back(rec);
                hProtonPt->Fill(p4.Pt());
            } else if (pid == -1120) {
                ev.antiProtons.push_back(rec);
                hProtonPt->Fill(p4.Pt());
            } else if (pid == 2130) {
                ev.lambdas.push_back(rec);
                hLambdaPt->Fill(p4.Pt());
            } else if (pid == -2130) {
                ev.antiLambdas.push_back(rec);
                hLambdaPt->Fill(p4.Pt());
            }
        }

        hNch->Fill(ev.nch);
        if (ev.nch < minNch) { // reject low multiplicity events
            continue;
        }

        acceptedEvents++; 
        totalSelectedProtons += ev.protons.size();
        totalSelectedAntiProtons += ev.antiProtons.size();
        totalSelectedLambdas += ev.lambdas.size();
        totalSelectedAntiLambdas += ev.antiLambdas.size();

        const bool hasP = (!ev.protons.empty() || !ev.antiProtons.empty());
        const bool hasLambda = (!ev.lambdas.empty() || !ev.antiLambdas.empty());
        const bool hasSameChargePLambda = ((!ev.protons.empty() && !ev.lambdas.empty()) || (!ev.antiProtons.empty() && !ev.antiLambdas.empty()));
        const bool hasPPLambda = ((ev.protons.size() >= 2 && !ev.lambdas.empty()) || (ev.antiProtons.size() >= 2 && !ev.antiLambdas.empty()));

        // updating the variables
        if (hasP) eventsWithP++;
        if (hasLambda) eventsWithLambda++;
        if (hasSameChargePLambda) eventsWithBothPLambda++;
        if (hasPPLambda) eventsWithPPLambda++;

        // Same-event p-Lambda.
        for (const auto& p : ev.protons) {                                           // paris of all lambdas and p
            for (const auto& l : ev.lambdas) {
                const double ks = calcKStar(p.p4, l.p4);
                if (ks >= 0.0 && std::isfinite(ks)) {
                    const double rLab = sourceSeparationLab(p, l);
                    const double rStar = sourceSeparationPairRestFrame(p, l);
                    hSame_pLambda_kstar->Fill(ks);
                    hSource_pLambda_rLab->Fill(rLab);
                    hSource_pLambda_rStar->Fill(rStar);
                    hKStarVsRStar_pLambda->Fill(ks, rStar);
                    samePairs++;
                }
            }
        }

        // Same-event anti-p anti-Lambda.
        for (const auto& p : ev.antiProtons) {
            for (const auto& l : ev.antiLambdas) {
                const double ks = calcKStar(p.p4, l.p4);
                if (ks >= 0.0 && std::isfinite(ks)) {
                    const double rLab = sourceSeparationLab(p, l);
                    const double rStar = sourceSeparationPairRestFrame(p, l);
                    hSame_pLambda_kstar->Fill(ks);
                    hSource_pLambda_rLab->Fill(rLab);
                    hSource_pLambda_rStar->Fill(rStar);
                    hKStarVsRStar_pLambda->Fill(ks, rStar);
                    samePairs++;
                }
            }
        }

        // Same-event ppLambda.
        for (std::size_t i = 0; i < ev.protons.size(); ++i) {
            for (std::size_t j = i + 1; j < ev.protons.size(); ++j) {
                for (const auto& l : ev.lambdas) {
                    const double q3 = Q3(ev.protons[i].p4, ev.protons[j].p4, l.p4);
                    if (q3 >= 0.0 && std::isfinite(q3)) {
                        const double rhoLab = tripletHyperradiusLab(ev.protons[i], ev.protons[j], l);
                        hSame_ppLambda_Q3->Fill(q3);
                        hSource_ppLambda_rhoLab->Fill(rhoLab);
                        sameTriplets++;
                    }
                }
            }
        }

        // Same-event anti-p anti-p anti-Lambda.
        for (std::size_t i = 0; i < ev.antiProtons.size(); ++i) {
            for (std::size_t j = i + 1; j < ev.antiProtons.size(); ++j) {
                for (const auto& l : ev.antiLambdas) {
                    const double q3 = Q3(ev.antiProtons[i].p4, ev.antiProtons[j].p4, l.p4);
                    if (q3 >= 0.0 && std::isfinite(q3)) {
                        const double rhoLab = tripletHyperradiusLab(ev.antiProtons[i], ev.antiProtons[j], l);
                        hSame_ppLambda_Q3->Fill(q3);
                        hSource_ppLambda_rhoLab->Fill(rhoLab);
                        sameTriplets++;
                    }
                }
            }
        }

        // Mixed-event p-Lambda.
        for (const auto& old : mixingBuffer) {
            for (const auto& p : ev.protons) {
                for (const auto& l : old.lambdas) {
                    const double ks = calcKStar(p.p4, l.p4);
                    if (ks >= 0.0 && std::isfinite(ks)) {
                        hMixed_pLambda_kstar->Fill(ks);
                        mixedPairs++;
                    }
                }
            }

            for (const auto& p : old.protons) {
                for (const auto& l : ev.lambdas) {
                    const double ks = calcKStar(p.p4, l.p4);
                    if (ks >= 0.0 && std::isfinite(ks)) {
                        hMixed_pLambda_kstar->Fill(ks);
                        mixedPairs++;
                    }
                }
            }

            for (const auto& p : ev.antiProtons) {
                for (const auto& l : old.antiLambdas) {
                    const double ks = calcKStar(p.p4, l.p4);
                    if (ks >= 0.0 && std::isfinite(ks)) {
                        hMixed_pLambda_kstar->Fill(ks);
                        mixedPairs++;
                    }
                }
            }

            for (const auto& p : old.antiProtons) {
                for (const auto& l : ev.antiLambdas) {
                    const double ks = calcKStar(p.p4, l.p4);
                    if (ks >= 0.0 && std::isfinite(ks)) {
                        hMixed_pLambda_kstar->Fill(ks);
                        mixedPairs++;
                    }
                }
            }
        }

        // Mixed-event ppLambda.
        // uses p from current event, p from oldA, and Lambda from oldB.
        if (mixingBuffer.size() >= 2) {
            for (std::size_t ia = 0; ia < mixingBuffer.size(); ++ia) {
                for (std::size_t ib = 0; ib < mixingBuffer.size(); ++ib) {
                    if (ia == ib) {
                        continue;
                    }
                    const auto& oldA = mixingBuffer[ia];
                    const auto& oldB = mixingBuffer[ib];
                    for (const auto& p1 : ev.protons) {
                        for (const auto& p2 : oldA.protons) {
                            for (const auto& l : oldB.lambdas) {
                                const double q3 = Q3(p1.p4, p2.p4, l.p4);
                                if (q3 >= 0.0 && std::isfinite(q3)) {
                                    hMixed_ppLambda_Q3->Fill(q3);
                                    mixedTriplets++;
                                }
                            }
                        }
                    }

                    for (const auto& p1 : ev.antiProtons) {
                        for (const auto& p2 : oldA.antiProtons) {
                            for (const auto& l : oldB.antiLambdas) {
                                const double q3 = Q3(p1.p4, p2.p4, l.p4);
                                if (q3 >= 0.0 && std::isfinite(q3)) {
                                    hMixed_ppLambda_Q3->Fill(q3);
                                    mixedTriplets++;
                                }
                            }
                        }
                    }
                }
            }
        }

        mixingBuffer.push_back(ev);
        if ((int) mixingBuffer.size() > mixDepth) {
            mixingBuffer.pop_front();
        }

        if ((iev + 1) % 10000 == 0) {
            std::cout<<"Processed "<<iev + 1 <<" / "<<maxEvents<<" events, accepted = "<<acceptedEvents <<", same pL pairs = "<<samePairs<<", mixed pL pairs = "<<mixedPairs<<", same ppL triplets = "<<sameTriplets<<", mixed ppL triplets = "<<mixedTriplets<<"\n";
        }
    }

    CorrelationC(hSame_pLambda_kstar, hMixed_pLambda_kstar, hCorr_pLambda_kstar, 2.0, 5.0);

    CorrelationC(hSame_ppLambda_Q3, hMixed_ppLambda_Q3, hCorr_ppLambda_Q3, 1.2, 2.4);

    std::cout<<"\nHistogram checks before writing:\n";
    std::cout<<"hSame_pLambda entries       = " <<hSame_pLambda_kstar->GetEntries() <<", integral with UF/OF = " <<hSame_pLambda_kstar->Integral(0, hSame_pLambda_kstar->GetNbinsX() + 1)<<"\n";
    std::cout<<"hMixed_pLambda entries      = "<<hMixed_pLambda_kstar->GetEntries()<<", integral with UF/OF = "<<hMixed_pLambda_kstar->Integral(0, hMixed_pLambda_kstar->GetNbinsX() + 1)<<"\n";
    std::cout<<"hSame_ppLambda entries      = "<<hSame_ppLambda_Q3->GetEntries()<<", integral with UF/OF = "<<hSame_ppLambda_Q3->Integral(0, hSame_ppLambda_Q3->GetNbinsX() + 1)<<"\n";
    std::cout<<"hMixed_ppLambda entries     = "<<hMixed_ppLambda_Q3->GetEntries()<<", integral with UF/OF = "<<hMixed_ppLambda_Q3->Integral(0, hMixed_ppLambda_Q3->GetNbinsX() + 1)<<"\n";
    std::cout<<"hSource_pLambda_rLab entries  = "<<hSource_pLambda_rLab->GetEntries()<<"\n";
    std::cout<<"hSource_pLambda_rStar entries = "<<hSource_pLambda_rStar->GetEntries()<<"\n";
    std::cout<<"hSource_ppLambda_rhoLab entries = "<<hSource_ppLambda_rhoLab->GetEntries()<<"\n";

    TFile out(outputFile, "RECREATE");

    hNch->Write();
    hProtonPt->Write();
    hLambdaPt->Write();

    hSame_pLambda_kstar->Write();
    hMixed_pLambda_kstar->Write();
    hCorr_pLambda_kstar->Write();

    hSame_ppLambda_Q3->Write();
    hMixed_ppLambda_Q3->Write();
    hCorr_ppLambda_Q3->Write();

    hSource_pLambda_rLab->Write();
    hSource_pLambda_rStar->Write();
    hSource_ppLambda_rhoLab->Write();
    hKStarVsRStar_pLambda->Write();

    out.Close();

    gSystem->mkdir("figures", kTRUE);
    gStyle->SetOptStat(0);

    savePlot(hNch, "figures/nch.png");
    savePlot(hProtonPt, "figures/proton_pt.png");
    savePlot(hLambdaPt, "figures/lambda_pt.png");
    savePlot(hSame_pLambda_kstar, "figures/same_pLambda_kstar.png");
    savePlot(hMixed_pLambda_kstar, "figures/mixed_pLambda_kstar.png");
    savePlot(hCorr_pLambda_kstar, "figures/corr_pLambda_kstar.png");
    savePlot(hSame_ppLambda_Q3, "figures/same_ppLambda_Q3.png");
    savePlot(hMixed_ppLambda_Q3, "figures/mixed_ppLambda_Q3.png");
    savePlot(hCorr_ppLambda_Q3, "figures/corr_ppLambda_Q3.png");
    savePlot(hSource_pLambda_rLab, "figures/source_pLambda_rLab.png");
    savePlot(hSource_pLambda_rStar, "figures/source_pLambda_rStar.png");
    savePlot(hSource_ppLambda_rhoLab, "figures/source_ppLambda_rhoLab.png");
    savePlot2D(hKStarVsRStar_pLambda, "figures/kstar_vs_rstar_pLambda.png");

    // Success print
    std::cout<<"\nDone.\n";
    std::cout<<"Accepted events                         : "<<acceptedEvents<<"\n";
    std::cout<<"Selected p                              : "<<totalSelectedProtons<<"\n";
    std::cout<<"Selected anti-p                         : "<<totalSelectedAntiProtons<<"\n";
    std::cout<<"Selected Lambda                         : "<<totalSelectedLambdas<<"\n";
    std::cout<<"Selected anti-Lambda                    : "<<totalSelectedAntiLambdas<<"\n";
    std::cout<<"Events with p or anti-p                 : "<<eventsWithP<<"\n";
    std::cout<<"Events with Lambda or anti-Lambda       : "<<eventsWithLambda<<"\n";
    std::cout<<"Events with same-channel pLambda        : "<<eventsWithBothPLambda<<"\n";
    std::cout<<"Events with same-channel ppLambda       : "<<eventsWithPPLambda<<"\n";
    std::cout<<"Same-event pLambda pairs                : "<<samePairs<<"\n";
    std::cout<<"Mixed-event pLambda pairs               : "<<mixedPairs<<"\n";
    std::cout<<"Same-event ppLambda triplets            : "<<sameTriplets<<"\n";
    std::cout<<"Mixed-event ppLambda triplets           : "<<mixedTriplets<<"\n";
    std::cout<<"Output ROOT file                        : "<<outputFile<<"\n";
    std::cout<<"Figures saved in                        : figures/\n\n";

    f->Close();
}