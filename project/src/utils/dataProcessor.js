import { Matrix } from 'ml-matrix';
import { PCA } from 'ml-pca';

export class DataProcessor {
    constructor() {
        this.pca = null;
    }

    async preprocessData(data) {
        // Normalize data
        const normalized = this.normalize(data);
        
        // Apply PCA for dimensionality reduction
        if (!this.pca) {
            this.pca = new PCA(normalized);
        }
        
        return this.pca.predict(normalized);
    }

    normalize(data) {
        const matrix = new Matrix(data);
        return matrix.scaleColumns();
    }
}