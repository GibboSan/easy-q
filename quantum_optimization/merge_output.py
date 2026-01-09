import os
import json
import argparse
import pandas as pd
from collections import OrderedDict

def collect_output_jsons(exp_path):
    all_rows = []
    column_order = None

    out_dirs = sorted([d for d in os.listdir(exp_path) if d.endswith("_out")])
    for d in out_dirs:
        json_path = os.path.join(exp_path, d, "output.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                try:
                    data = json.load(f, object_pairs_hook=OrderedDict)
                except json.JSONDecodeError:
                    print(f"⚠️  JSON non valido in: {json_path}")
                    continue

                row = OrderedDict()
                row["label"] = d  # Prima colonna

                for k, v in data.items():
                    # Bitstring o frequency: forzare a stringa
                    if isinstance(v, list):
                        row[k] = json.dumps(v)
                    elif isinstance(v, (int, float)) and "bitstring" in k:
                        row[k] = str(v).zfill(len(str(v)))  # Pad con zeri iniziali se serve
                    elif "bitstring" in k or "frequency" in k:
                        row[k] = str(v)
                    else:
                        row[k] = v

                all_rows.append(row)

    # Normalize nested dicts to flatten them
    if all_rows:
        df = pd.json_normalize(all_rows)
        # Move 'label' to first column
        cols = ['label'] + [col for col in df.columns if col != 'label']
        df = df[cols]
        return df.to_dict('records'), list(df.columns)
    else:
        return [], None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_path", help="Path alla cartella dell'esperimento (es: exp_20250630_101112)")
    args = parser.parse_args()

    rows, columns = collect_output_jsons(args.exp_path)

    if not rows:
        print("❌ Nessun output.json trovato.")
        return

    df = pd.DataFrame(rows)
    df = df[columns]  # Ordina le colonne

    csv_path = os.path.join(args.exp_path, "merged_results.csv")
    xlsx_path = os.path.join(args.exp_path, "merged_results.xlsx")

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False, engine="openpyxl")

    print(f"✅ File CSV creato: {csv_path}")
    print(f"✅ File Excel creato: {xlsx_path}")

if __name__ == "__main__":
    main()
