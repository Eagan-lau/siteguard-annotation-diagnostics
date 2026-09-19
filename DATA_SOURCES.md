# Data sources

Provider releases used in the study. Third-party terms apply to the source records and pretrained models.

| Resource | Release or snapshot | Role | Download route |
| --- | --- | --- | --- |
| UniProtKB/Swiss-Prot | 2026_02 | sequences, EC/Rhea annotations, evidence and catalytic-site records | <https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/> |
| UniProtKB/Swiss-Prot | 2026_01 | temporal T1 reference | <https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2026_01/knowledgebase/uniprot_sprot-only2026_01.tar.gz> |
| UniProtKB/Swiss-Prot | 2023_01 | temporal T0 reference | <https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2023_01/knowledgebase/uniprot_sprot-only2023_01.tar.gz> |
| UniProt ID Mapping | snapshot 2026-08-18 | current metadata for a 50,000-accession panel | <https://rest.uniprot.org/idmapping/run> |
| Rhea and release-matched ChEBI mappings | 141, 140 and 126 | exact reaction vocabulary and temporal mappings | <https://ftp.expasy.org/databases/rhea/old_releases/> |
| ExPASy ENZYME | 2026-06-10 | EC canonicalization and status | <https://ftp.expasy.org/databases/enzyme/> |
| M-CSA | snapshot 2026-08-18 | curated catalytic residues and reference structures | <https://www.ebi.ac.uk/thornton-srv/m-csa/api/entries/?format=json> |
| AlphaFold Protein Structure Database | Swiss-Prot v6; snapshot 2026-08-18 | predicted structures, pLDDT and selected PAE | <https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_cif_v6.tar> |
| Protein Data Bank | enzyme-related mmCIF snapshot 2026-08-18 | experimental structures and external cohorts | <https://files.rcsb.org/download/> |
| SIFTS | snapshot 2026-08-18 | PDB-UniProt and residue mapping | <https://ftp.ebi.ac.uk/pub/databases/msd/sifts/> |
| PDB Chemical Component Dictionary | snapshot 2026-08-18 | ligand and modified-residue normalization | <https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz> |
| Pfam | release observed 2026-01-22 | domain architecture and family analyses | <https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/> |
| CATH | 4.4.0 | structural classification | <https://download.cathdb.info/cath/releases/all-releases/v4_4_0/> |
| P450Rdb | v2 snapshot 2026-08-18 | cytochrome P450 evaluation | <https://www.cellknowledge.com.cn/p450rdb_v2/download.html> |
| PlantP450 | snapshot 2026-08-18 | cytochrome P450 evaluation | <https://erda.dk/public/vgrid/PlantP450/> |
| FunP450 | snapshot 2026-08-18 | cytochrome P450 evaluation | <https://p450.biodesign.ac.cn/> |
| CYP nomenclature | snapshot 2026-08-18 | family/subfamily normalization | <https://drnelson.uthsc.edu/nomenclature/> |
| PubChem | targeted 2,719-CID snapshot 2026-08-18 | P450 compound normalization | <https://pubchem.ncbi.nlm.nih.gov/rest/pug/> |
| ESM2 | `esm2_t33_650M_UR50D` and `esm2_t12_35M_UR50D` | protein and residue representations | <https://huggingface.co/facebook/esm2_t33_650M_UR50D> |


CLEAN code and model resources: https://github.com/tttianhao/CLEAN. The paired comparison uses the saved outputs provided in the accompanying data archive.

## Native and temporal extensions

Native CLEAN/DIAMOND inputs and provenance are in `data/native_workflows/`. The unchanged-sequence temporal comparison uses official Swiss-Prot 2026_02 and 2026_03 XML releases from https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/. Source checksums, accession-level evidence and screening counts are in `data/temporal_pilot/`. Original provider licenses remain applicable.
