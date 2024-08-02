# Cell Lifelines Autoencoder

![Conference Poster](Conference%20Poster.png)

## Abstract

This project applies deep learning and latent space analysis to study cell behavior in bioreactors using Computational Fluid Dynamics (CFD) data. We employ LSTM-based autoencoders to reduce the dimensionality of metabolic data, followed by clustering and feature importance analysis to identify crucial metabolic states. Our approach combines perturbation analysis and SHAP values to enhance interpretability, improving our understanding of cell heterogeneity and identifying key markers for bioreactor performance evaluation.

## Methodology

1. **Autoencoder Utilization**: LSTM autoencoders reduce high-dimensional metabolic data from CFD and dFBA of Textbook Model E. Coli.
2. **Data Reduction**: Compress 3D input data (lifelines, timepoints, reaction fluxes) into a lower-dimensional latent space.
3. **Interpretability Enhancement**: Use perturbation analysis and SHAP values to clarify latent space feature importance.
4. **Clustering**: Apply t-SNE embeddings & k-means clustering to identify key metabolic states at each timestep.

## Key Findings

- **Dimension Reduction**: Achieved 80% reduction with minimal information loss.
- **Latent Space Interpretation**: Perturbation analysis revealed identifiable metabolic states.
- **Clustering**: t-SNE embeddings visualized distinct metabolic states.
- **Feature Importance**: Flux through the Krebs cycle emerged as a critical factor in codifying metabolic states.
- **Cell Lifeline Analysis**: Identified clusters of cells with similar metabolic states over time, revealing potential suboptimal conditions.

## Analysis Highlights

1. **Autoencoder Reconstruction**: Significantly reduced, indicating improved feature representation accuracy.
2. **Perturbation Analysis**: Showcases the effect of zeroing latent space neurons on reconstruction.
3. **Labeled Biomass Reconstruction**: Visualizes biomass reaction over time with labeled metabolic states.
4. **Discrepancy Heatmap**: Highlights significant neurons impacting reconstruction.
5. **SHAP Feature Importance**: Identifies key features affecting model predictions (e.g., CO2t, GLUDy, PGK, PGM, Ex_nh4_e, SUCOAS, H2Ot, Ex_o2_e).
6. **TSNE Embedding Animation**: Visualizes latent space evolution over time, showing clustering and transitions.

## Conclusion

Our approach effectively reduces data dimensionality while maintaining interpretability, offering valuable insights into cell behavior in bioreactors. The identified metabolic states and important features provide a foundation for optimizing bioreactor performance and understanding cellular heterogeneity.
