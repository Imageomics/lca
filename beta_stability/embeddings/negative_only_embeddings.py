import functools
import pandas as pd
from beta_stability.embeddings.embeddings import Embeddings


class NegativeOnlyEmbeddings(Embeddings):
    """
    Embeddings wrapper that filters based on tracking id.
    Returns 1.0 for the same tracking ids, original score for others (optionally multiplied by a constant).
    """
    
    def __init__(self, 
                 embeddings,
                 ids,
                 data_df, 
                 node2uuid, 
                 distance_power=1,
                 print_func=print,
                 id_key='uuid', 
                 class_key='class_id', 
                 multiplier=1.0):
        # Initialize parent class
        super().__init__(embeddings, ids, distance_power, print_func)
        
        # Additional attributes for tracking ID filtering
        self.data_df = data_df
        self.node2uuid = node2uuid
        self.id_key = id_key
        self.class_key = class_key
        self.multiplier = multiplier

        print_func(f"Created NegativeOnlyEmbeddings with {len(self.embeddings)} embeddings")
    
    def get_base(self):
        """Return the base embeddings without filtering."""
        return Embeddings(self.embeddings, self.uuids, self.distance_power, self.print_func)

    @classmethod
    def from_embeddings(cls, 
                        base_embeddings,
                        data_df,
                        node2uuid,
                        id_key='uuid',
                        class_key='class_id',
                        multiplier=1.0):
        """Create NegativeOnlyEmbeddings from an existing Embeddings instance."""
        base_embeddings.print_func(f"Created NegativeOnlyEmbeddings with base {type(base_embeddings).__name__}")
        return cls(
            embeddings=base_embeddings.embeddings,
            ids=base_embeddings.uuids,
            data_df=data_df,
            node2uuid=node2uuid,
            distance_power=base_embeddings.distance_power,
            print_func=base_embeddings.print_func,
            id_key=id_key,
            class_key=class_key,
            multiplier=multiplier
        )
    
    def _is_plausible(self, n0, n1):
        """Check if edge is plausible based on class id."""
        uuid1 = self.node2uuid[n0]
        uuid2 = self.node2uuid[n1]
        class_id1 = self.data_df.loc[self.data_df[self.id_key] == uuid1, self.class_key].squeeze()
        class_id2 = self.data_df.loc[self.data_df[self.id_key] == uuid2, self.class_key].squeeze()
        return class_id1 == class_id2

    def get_score(self, n0, n1):
        """Return 1.0 if same class id, base score if different."""
        if self._is_plausible(n0, n1):
            return 1.0
        else:
            # Use parent's get_score method
            return super().get_score(n0, n1) * self.multiplier
    
    def _apply_metadata_filter_to_distmat(self, distmat, ids, start_idx):
        """Apply metadata filtering to distance matrix by setting implausible edges to infinity."""
        for i in range(distmat.shape[0]):
            for j in range(distmat.shape[1]):
                if i + start_idx != j:  # Don't filter self-edges
                    id1, id2 = ids[i + start_idx], ids[j]
                    if self._is_plausible(id1, id2):
                        distmat[i, j] = 0
                    else:
                        distmat[i, j] = 1 - self.multiplier + self.multiplier * distmat[i, j]
        return distmat
    
    @functools.cache
    def _calculate_distance_matrix(self, flags=None):
        # Call parent's _calculate_distance_matrix to get chunks
        chunks, ids = super()._calculate_distance_matrix(flags=flags)
        
        # Apply tracking ID filtering to all chunks
        filtered_chunks = []
        start = 0
        for chunk in chunks:
            filtered_chunk = self._apply_metadata_filter_to_distmat(chunk.copy(), ids, start)
            filtered_chunks.append(filtered_chunk)
            start += chunk.shape[0]
        
        return filtered_chunks, ids

