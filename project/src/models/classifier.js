import * as tf from '@tensorflow/tfjs';

export class NetworkClassifier {
    constructor(numClasses = 8) {
        this.numClasses = numClasses;
        this.model = this.buildModel();
        this.threshold = 0.8;
    }

    buildModel() {
        const model = tf.sequential();
        
        // CNN layers
        model.add(tf.layers.reshape({
            targetShape: [80, 1],
            inputShape: [80]
        }));
        
        model.add(tf.layers.conv1d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu'
        }));
        
        // LSTM layer
        model.add(tf.layers.lstm({
            units: 128,
            returnSequences: true
        }));
        
        // Global pooling
        model.add(tf.layers.globalAveragePooling1d());
        
        // Dense layers
        model.add(tf.layers.dense({
            units: this.numClasses,
            activation: 'softmax'
        }));
        
        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        return model;
    }

    async predict(data) {
        const tensorData = tf.tensor2d(data, [-1, 80]);
        const predictions = this.model.predict(tensorData);
        const warnings = tf.max(predictions, 1).greater(this.threshold);
        const result = {
            predictions: await predictions.array(),
            warnings: await warnings.array()
        };
        tensorData.dispose();
        predictions.dispose();
        warnings.dispose();
        return result;
    }
}