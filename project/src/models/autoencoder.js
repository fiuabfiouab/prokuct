import * as tf from '@tensorflow/tfjs';

export class AutoEncoder {
    constructor(inputDim = 80, encodingDim = 32) {
        this.inputDim = inputDim;
        this.encodingDim = encodingDim;
        this.model = this.buildModel();
    }

    buildModel() {
        const model = tf.sequential();
        
        // Encoder
        model.add(tf.layers.dense({
            units: this.encodingDim,
            activation: 'relu',
            inputShape: [this.inputDim]
        }));
        
        // Decoder
        model.add(tf.layers.dense({
            units: this.inputDim,
            activation: 'sigmoid'
        }));
        
        model.compile({
            optimizer: 'adam',
            loss: 'meanSquaredError'
        });
        
        return model;
    }

    async train(data, epochs = 10) {
        const tensorData = tf.tensor2d(data);
        await this.model.fit(tensorData, tensorData, {
            epochs,
            batchSize: 32,
            shuffle: true
        });
        tensorData.dispose();
    }

    encode(data) {
        const tensorData = tf.tensor2d(data);
        const encoded = this.model.layers[0].apply(tensorData);
        tensorData.dispose();
        return encoded;
    }
}