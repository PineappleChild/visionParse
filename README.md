batch of images -> DINOv2(for feature extraction) -> clustering with Kmeans -> score clusters with CLIP and DINOv2 similarity between classification(CLS) tokens and patch features -> (For now visualize CLIP classification, DINOv2 cls and patch alignment, PCA visualization of DINOv2 features) 

want to:
-improve clustering, lots of artifacts and blending between background and targets causing poor results
-combine scores to better detect if a cluster is background or part of the target
-use score and SAM to segment the inputted images
-mask the background so only the target is in the image(going to use images for downstream training for anomaly detection)


future:
automatically lable each cluster with no supervision(currently need to feed clip prompts for target and background)
to reach goal of having a fully self supervised pipeline for lableing and segmenting an images

could be cool to use this with some data gathering tool to automatically create datasets for future projects
