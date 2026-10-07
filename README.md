# EC-DynaMech v0.1

Werkende lokale PWA-prototype voor iPhone, iPad en desktop.

## Starten

Open deze map via een lokale of gehoste webserver. Voor een snelle lokale test:

```bash
python3 -m http.server 8000
```

Open daarna `http://localhost:8000/ec_dynamech_v0_1/`.

Op iOS: open het gehoste adres in Safari en kies **Deel > Zet op beginscherm**.

## Inhoud

- canonieke R9-topologie
- vijf instelbare operatoren: rho, mu, iota, C en Pi
- vijf afzonderlijke operatorvelden
- samengesteld `L_DG`
- emergente isobanden en kandidaat-`psi_L`-punten
- prototype-indicatoren voor sluiting, ruit, Psi en Q
- JSON-export
- offline cache na eerste succesvolle opening

## Methodologische status

`L_DG` is in v0.1 een genormaliseerde, afgeleide organisatie-intensiteit. Het is geen fysieke luminantie. Alle uitkomsten zijn computationeel en niet onafhankelijk gevalideerd. De operatorwaarden zijn simulatie-invoer; latere versies moeten meetdefinities, onzekerheid en empirische validatie toevoegen.
