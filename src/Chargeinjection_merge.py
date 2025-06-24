import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# ====== Paths and filenames ======
base_dir = "./"  # Adjust this if needed
map_file = os.path.join(base_dir, "map.csv")
real_data_path = os.path.join(base_dir, "analysis_files")
real_data = {
    "Electron_0p5GeV": "fixed_Electron_0.5GeV.csv",
    "Muon_4GeV": "fixed_Muon_4GeV.csv",
    "Electron_4GeV": "fixed_Electron_4GeV.csv",
}
TOT_ADC_pdf = os.path.join(base_dir, "TOT_ADC_Calibration_AllChannels.pdf")

# ====== Helper functions ======
def linear_asymptote(x, a, b): return a * x + b
def func(x, a, b, p0, p1): return p0*x + p1 - a/(x - b)
def tot_to_charge_func(tot, TOTs, charges):
    for i in range(len(TOTs)):
        if TOTs[i] >= tot:
            return charges[i]
    return charges[-1]
def charge_to_adc_func(charge, linearfitcoeffs):
    return linearfitcoeffs[0]*charge + linearfitcoeffs[1]

# ====== Load map file ======
map_df = pd.read_csv(map_file)

# ====== Open PDF writer ======
with PdfPages(TOT_ADC_pdf) as pdf:
    for idx, row in map_df.iterrows():
        try:
            layer, strip, end = int(row["PLANE"]), int(row["STRIP"]), int(row["END"])
            dpm, ilink, channel = int(row["DPM"]), int(row["LINK"]), int(row["CHAN"])

            if dpm == 0:
                filename = os.path.join(base_dir, "scan_DPM0_CALIBRUN_coff14_20220424_220633.csv")
            else:
                filename = os.path.join(base_dir, "scan_DPM1_CALIBRUN_coff14_20220424_220543.csv")

            with open(filename, "r") as file:
                events = []
                for line in file:
                    s = line.strip().split(",")
                    if s[0] == "CALIB_DAC":
                        continue
                    CALIB_DAC = int(s[0])
                    DPM = int(s[1])
                    ILINK = int(s[2])
                    CHAN = int(s[3])
                    if DPM != dpm or ILINK != ilink or CHAN != channel:
                        continue
                    ADC = [int(s[i]) for i in range(5, 13)]
                    TOT = [int(s[i]) for i in range(13, 21)]
                    CAPACITOR_TYPE = s[29]
                    sumADC = sum(ADC)
                    maxADC = max(ADC)
                    TOTsum = sum(TOT)
                    events.append({
                        "calib_dac": CALIB_DAC,
                        "capacitor_type": CAPACITOR_TYPE,
                        "sumADC": sumADC,
                        "maxADC": maxADC,
                        "TOT": TOTsum
                    })

            low = [e for e in events if e["capacitor_type"] == "0"]
            high = [e for e in events if e["capacitor_type"] == "1"]

            def filter_events(evts, tot_threshold=40, adc_threshold=40):
                result = []
                for cd in set(e["calib_dac"] for e in evts):
                    group = [e for e in evts if e["calib_dac"] == cd]
                    if len(group) < 3:
                        continue
                    sadcs = [e["sumADC"] for e in group]
                    tots = [e["TOT"] for e in group]
                    if (max(sadcs)-min(sadcs) < adc_threshold and max(tots) == 0) or (max(tots)-min(tots) < tot_threshold):
                        avg = lambda k: sum(e[k] for e in group) / len(group)
                        result.append({
                            "calib_dac": cd,
                            "sumADC": avg("sumADC"),
                            "maxADC": avg("maxADC"),
                            "TOT": avg("TOT"),
                            "capacitor_type": group[0]["capacitor_type"]
                        })
                return result

            lowf = filter_events(low)
            highf = filter_events(high, tot_threshold=20)

            for e in low + lowf:
                e["charge"] = e["calib_dac"] / 2048 * 500
            for e in high + highf:
                e["charge"] = e["calib_dac"] / 2048 * 8000

            linearfit = np.polyfit([x["charge"] for x in lowf], [x["sumADC"] for x in lowf], 1)
            leadfit = np.polyfit([x["charge"] for x in highf if x["calib_dac"] < 100], [x["sumADC"] for x in highf if x["calib_dac"] < 100], 1)

            a, b = leadfit
            c, d = linearfit
            for e in high + highf:
                e["charge"] = a/c * e["charge"] + (b - d)/c

            TOTthreshold = min([e["charge"] for e in highf if e["TOT"] > 0])
            xval = [e["charge"] for e in highf if e["charge"] >= TOTthreshold]
            yval = [e["TOT"] for e in highf if e["charge"] >= TOTthreshold]
            popt, _ = curve_fit(linear_asymptote, xval[-5:], yval[-5:])
            poptpower, _ = curve_fit(lambda x, a, b: func(x, a, b, *popt), xval, yval, bounds=([-10000, 0], [10000, TOTthreshold - 10]))

            charges = np.linspace(TOTthreshold, 8*2047, 1000)
            TOTs = [func(p, *poptpower, *popt) for p in charges]
            sumADCs = [linearfit[0]*p + linearfit[1] for p in charges]

            threshold = min([x["TOT"] for x in highf if x["charge"] >= TOTthreshold])
            for p in highf:
                if p["TOT"] >= threshold:
                    t = p["TOT"]
                    for i in range(len(TOTs)):
                        if TOTs[i] >= t:
                            p["sumADC"] = sumADCs[i]
                            break

            # ===== Page 1: Title =====
            plt.figure()
            plt.axis('off')
            plt.title(f"Calibration - Layer {layer}, Strip {strip}, End {end}", fontsize=18)
            pdf.savefig()
            plt.close()

            # ===== Page 2: Figure 1 =====
            plt.figure()
            plt.plot(np.linspace(min([x["charge"] for x in lowf]), max([x["charge"] for x in highf]), 2),
                     [linearfit[0]*p + linearfit[1] for p in np.linspace(min([x["charge"] for x in lowf]), max([x["charge"] for x in highf]), 2)],
                     label="Fit to low cap. region", linestyle="--", color="black")
            plt.scatter([x["charge"] for x in lowf], [x["sumADC"] for x in lowf], label="Low Cap.", color="blue")
            plt.scatter([x["charge"] for x in highf if x["charge"] < TOTthreshold], [x["sumADC"] for x in highf if x["charge"] < TOTthreshold], label="High Cap.", color="green")
            plt.scatter([x["charge"] for x in highf if x["charge"] >= TOTthreshold], [x["sumADC"] for x in highf if x["charge"] >= TOTthreshold], label="High Cap. (TOT)", marker="x")
            plt.axvline(TOTthreshold, color="red", label="TOT Threshold")
            plt.xlabel("Charge [fC]")
            plt.ylabel("Sum of ADC")
            plt.legend()
            pdf.savefig()
            plt.close()

            # ===== Page 3: Figure 2 =====
            plt.figure()
            x = np.linspace(min(xval), max(xval), 100)
            plt.plot(x, [func(p, *poptpower, *popt) for p in x], label="TOT fit", linestyle="--")
            plt.plot(x, [linear_asymptote(p, *popt) for p in x], label="Asymptote", color="green")
            plt.scatter(xval, yval, label="High Cap. TOT", color="black", marker="x")
            plt.xlabel("Charge [fC]")
            plt.ylabel("TOT")
            plt.legend()
            pdf.savefig()
            plt.close()

            # ===== Page 4: Figure 3 (Raw TOT) =====
            plt.figure()
            plt.scatter([x["charge"] for x in high if x["charge"] >= TOTthreshold], [x["TOT"] for x in high if x["charge"] >= TOTthreshold], color="green")
            plt.xlabel("Charge [fC]")
            plt.ylabel("TOT")
            plt.title("Raw TOT vs Charge")
            pdf.savefig()
            plt.close()

            # ===== Page 5–7: Real Data (Muon, e0.5, e4) =====
            for label in ["Muon_4GeV", "Electron_0p5GeV", "Electron_4GeV"]:
                fname = real_data[label]
                df = pd.read_csv(os.path.join(real_data_path, fname))
                df = df[(df["layer"] == layer) & (df["strip"] == strip)]
                tot_col = df[f"tot_end{end}"]
                adc_col = df[f"adc_sum_end{end}"]
                hist, bins = np.histogram(tot_col[tot_col > 0], bins=np.linspace(0, 5000, 200))
                new_threshold = next((bins[i] for i in range(len(hist)) if hist[i] >= 3), 0)

                adc_list, tot_adc = [], []
                for t, a in zip(tot_col, adc_col):
                    if t < new_threshold:
                        adc_list.append(a)
                    else:
                        charge = tot_to_charge_func(t, TOTs, charges)
                        adc_equiv = charge_to_adc_func(charge, linearfit)
                        tot_adc.append(adc_equiv)

                plt.figure()
                plt.hist([adc_list, tot_adc], bins=np.linspace(0, 25000, 40),
                         label=[f"{label} - ADC", f"{label} - TOT→ADC"], stacked=True, alpha=0.7)
                plt.yscale("log")
                plt.xlabel("Sum of ADC")
                plt.ylabel("Events")
                plt.title(f"{label} TOT to ADC")
                plt.legend()
                pdf.savefig()
                plt.close()

        except Exception as e:
            print(f"[Skipped] Layer {layer} Strip {strip} End {end}: {e}")
            continue
