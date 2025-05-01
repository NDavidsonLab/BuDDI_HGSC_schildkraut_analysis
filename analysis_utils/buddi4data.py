class BuDDI4Data:
    def __init__(self, 
                 X_unkp, y_unkp, 
                 label_unkp, drug_unkp, 
                 bulk_unkp, meta_unkp,
                 X_kp, y_kp, 
                 label_kp, drug_kp, 
                 bulk_kp, meta_kp,
                 gene_names, cell_type_names, encode_meta):
        
        """
        Data class for BuDDI dataset.

        :param X_unkp: Feature matrix for unknown proportions
        :param y_unkp: Target matrix for unknown proportions
        :param label_unkp: Labels for unknown proportions
        :param drug_unkp: Drug labels for unknown proportions
        :param bulk_unkp: Bulk labels for unknown proportions
        :param meta_unkp: Metadata for unknown proportions
        :param X_kp: Feature matrix for known proportions
        :param y_kp: Target matrix for known proportions
        :param label_kp: Labels for known proportions
        :param drug_kp: Drug labels for known proportions
        :param bulk_kp: Bulk labels for known proportions
        :param meta_kp: Metadata for known proportions
        :param gene_names: List of gene names
        :param cell_type_names: List of cell type names
        :param encode_meta: Boolean indicating whether to encode metadata
        """
        
        self.data = {
            "unkp": {"X": X_unkp, "y": y_unkp, "label": label_unkp, "drug": drug_unkp, "bulk": bulk_unkp, "meta": meta_unkp},
            "kp":   {"X": X_kp,   "y": y_kp,   "label": label_kp,   "drug": drug_kp,   "bulk": bulk_kp,   "meta": meta_kp}
        }
        self._gene_names = gene_names
        self._cell_type_names = cell_type_names
        self._encode_meta = encode_meta

    def __len__(self):
        return sum(len(self.data[key]["X"]) for key in self.data if "X" in self.data[key])

    def __repr__(self):
        return (f"BuDDIData(unkp_samples={len(self.data['unkp']['X'])}, "
                f"kp_samples={len(self.data['kp']['X'])}, "
                f"genes={len(self.gene_names)}, "
                f"cell_types={len(self.cell_type_names)})")

    def get(self, condition, key):
        return self.data.get(condition, {}).get(key, None)
    
    @property
    def gene_names(self):
        return self._gene_names
    @property
    def cell_type_names(self):
        return self._cell_type_names
    @property
    def encode_meta(self):
        return self._encode_meta