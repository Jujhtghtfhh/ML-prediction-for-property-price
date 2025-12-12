Sistema, kuri prognozuoja nekilnojamojo turto kainą pagal:
- Vietą
- Turto tipą  
- Miegamųjų ir vonios kambarių skaičių
- Vidaus ir išorės plotą
- Papildomus požymius (pvz., "Close to Town", "Private Pool")
- Įkeltas nuotraukas

0. **Įdiekite duomenis:**
 Pirma atsisiūskite duomenis
 https://zenodo.org/records/10962212

 Tada išpakuokite descriptions bei images folderius
 Projekto direktorijoj sukurkite data bei static folderius
 Tada i data folderi sudekite savo duomenis (descriptions images properties)
 
 ./data: 
  - /decriptions 
  - /images 
  - properties.csv

1. **Įdiekite priklausomybes:**
```bash
pip install flask pandas scikit-learn numpy pillow
```

2. **Apmokykite modelį (jei turite duomenis):**
```bash
python train_model_large.py
```

3. **Paleiskite serverį:**
```bash
python app.py
```

4. **Atidarykite naršyklėje:**
```
http://localhost:5000
```

---

## Mašininio mokymosi modelis

### Naudojami algoritmai:
- **Random Forest Regressor** - pagrindinis prognozavimo modelis
- **Standard Scaler** - požymių normalizavimas
- **Label Encoder** - kategorinių kintamųjų kodavimas

### Požymiai (Features):

**Skaitiniai:**
- Miegamųjų skaičius
- Vonios kambarių skaičius
- Vidaus plotas (m²)
- Išorės plotas (m²)

**Kategoriniai:**
- Vieta (Coín, Estepona, Marbella, etc.)
- Turto tipas (Villa, Apartment, Finca, etc.)

**Binariniai požymiai:**
- Close to Shops, Mountain Views, Private Garden, Private Pool, Sea Views, Beachfront, ir kt.

**Vaizdo požymiai (iš nuotraukų):**
- Spalvų statistika (RGB vidurkiai, std)
- Ryškumas
- Spalvų proporcijos
- Kraštų gradientai

---

## Duomenų formatas (CSV)

```csv
reference,location,price,title,bedrooms,bathrooms,indoor_area,outdoor_area,features
R31352,"Coín, Costa del Sol","€115,000",Plot,625,NA,NA,NA,Close to Shops|Mountain Views|Private Garden
R20329,"Estepona, Costa del Sol","€699,000",5 Bedroom Finca,5,2,250,400,Mountain Views|Private Pool|Sea Views
```
