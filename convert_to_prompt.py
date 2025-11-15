import json

with open("dane.json", "r") as f:
    records = json.load(f)

with open("dane_prompt.jsonl", "w") as out:
    for row in records:
        rozmowa = "\n".join(row["rozmowa"])
        przyczyny = ", ".join([
            "BD", "Zasilacz", "Tłumienie", "Linia światłowodowa", "Router", "Radiówka", "Core", "Sprzet",
            "Po stronie klienta", "Kabel", "SGT", "Aktywacja", "Konfiguracja", "Awaria globalna", "EPIX",
            "WiFi", "Zapytanie", "Sieć energetyczna", "Zmiana hasła", "tSEC", "BOK", "Dubel", "Odwołanie"
        ])
        prompt = (
            f"ID zgłoszenia: {row['id_zgloszenia']}\n"
            f"ID klienta: {row['id_klienta']}\n"
            f"Temat: {row['temat']}\n"
            f"Rozmowa:\n{rozmowa}\n\n"
            f"Możliwe przyczyny: {przyczyny}\n\n"
            f"Wybierz jedną przyczynę z listy powyżej i wypisz tylko ją (np. \"Zasilacz\"):"
        )
        completion = row["przyczyna"]
        json.dump({"prompt": prompt, "completion": f" {completion}"}, out)
        out.write("\n")
