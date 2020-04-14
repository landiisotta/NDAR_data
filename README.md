# NDAR data stratification

### Technical requirements
```buildoutcfg
Python 3.6+
```

The full list of required Python Packages is available in `requirements.txt` file. 
All the dependencies can be installed by:
```buildoutcfg
pip install -r requirements.txt
```

# Pipeline modules
### Dataset
Included so far:

- Autism Diagnostic Observation Schedule, 2nd Edition (ADOS-2) - Module Toddler;
- Autism Diagnostic Observation Schedule, 2nd Edition (ADOS-2) - Module 1, 2, 3, 4;
- Social Responsiveness Scale (SRS);
- Vineland-II - Survey Form (2005);
- Vineland-II - Parent and Caregiver Rating Form (2005)
- Mullen Scales of Early Learning;

### Codebook requirements
For every instrument, the codebook file should include the columns:
> `NDARname` with the level 1 item names from the NDAR database;

> `max_score` with the maximum score for each item;

> `subscale` with the level 2 subscale to which each item belongs;

> `scale` with the level 3 general content area (e.g., social communication/interaction).


