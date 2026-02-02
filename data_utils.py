from torch_geometric.utils import to_dense_batch


# ----------------------------------------------------
#    UNPACK BATCH TO REUSE FUNCTIONS
# ----------------------------------------------------
def unpack_batch(batch):
    """
    Supports both:
    - PyG ModelNet batches (Data object with .pos and .y)
    - ShapeNet batches (tuple: points, object_class, seg_labels, num_seg_classes)
    Returns:
      x: [B, N, 3]
      y: [B]
    """
    # ---------- PyG / ModelNet ----------
    if hasattr(batch, "pos") and hasattr(batch, "y"):
        # Pointnet needs: [object, nPoints, coordinades] 
        # i.e. [32 object, 1024 points, 3 coordinates]: [batch_size, nPoints, 3]
        # Since batch.batch is [32x1024, 3] we have to split it into individual object points.
        # It could be done manually but "to_dense_batch" does that.
        # "to_dense_batch" will also add padding if not all objects have same number of points. In our case they have.
        # mask is a [batch_size, nPoints] boolean saying if an entry is actually a real point or padding
        BatchPointsCoords, _ = to_dense_batch(batch.pos, batch.batch)  
        labels = batch.y                                                 
    
        return BatchPointsCoords, labels

    # ---------- ShapeNet ----------
    elif isinstance(batch, (tuple, list)):
        points, object_class, _, _ = batch
        BatchPointsCoords = points.transpose(-2,-1)
        labels = object_class

        return BatchPointsCoords, labels

    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

