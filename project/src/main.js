import * as tf from '@tensorflow/tfjs';
import { AutoEncoder } from './models/autoencoder.js';
import { NetworkClassifier } from './models/classifier.js';
import { RealTimeMonitor } from './models/monitor.js';
import { DataProcessor } from './utils/dataProcessor.js';
import { MetricsTracker } from './utils/metrics.js';

async function main() {
    await tf.ready();
    console.log('TensorFlow.js initialized');
    
    const dataProcessor = new DataProcessor();
    const metricsTracker = new MetricsTracker();
    
    // Initialize models
    const autoencoder = new AutoEncoder();
    const classifier = new NetworkClassifier();
    const monitor = new RealTimeMonitor(classifier);
    
    // Training data (simulated for demonstration)
    const numSamples = 1000;
    const trainingData = tf.randomNormal([numSamples, 80]);
    const labels = tf.randomUniform([numSamples, 8]);
    
    console.log('Starting training...');
    
    // Training loop
    const epochs = 50;
    const batchSize = 32;
    
    for (let epoch = 0; epoch < epochs; epoch++) {
        const batchCount = Math.floor(numSamples / batchSize);
        let epochLoss = 0;
        let predictions = [];
        
        for (let batch = 0; batch < batchCount; batch++) {
            const start = batch * batchSize;
            const end = start + batchSize;
            
            const batchData = trainingData.slice([start, 0], [batchSize, -1]);
            const batchLabels = labels.slice([start, 0], [batchSize, -1]);
            
            const result = await classifier.model.trainOnBatch(batchData, batchLabels);
            epochLoss += result;
            
            // Get predictions for metrics
            const batchPreds = classifier.model.predict(batchData);
            predictions.push(...Array.from(batchPreds.arraySync()));
        }
        
        // Update metrics
        metricsTracker.updateMetrics(
            epoch,
            epochLoss / batchCount,
            predictions,
            Array.from(labels.arraySync())
        );
        
        console.log(`Epoch ${epoch + 1}/${epochs}`);
        console.log(`Loss: ${(epochLoss / batchCount).toFixed(4)}`);
        console.log(`Accuracy: ${metricsTracker.accuracies[epoch].toFixed(4)}`);
        console.log(`F-Score: ${metricsTracker.fScores[epoch].toFixed(4)}`);
        console.log('-------------------');
    }
    
    // Save metrics to file
    const metrics = {
        epochs: metricsTracker.epochs,
        losses: metricsTracker.losses,
        accuracies: metricsTracker.accuracies,
        fScores: metricsTracker.fScores
    };
    
    console.log('Training completed!');
    console.log('Final metrics:', {
        loss: metrics.losses[metrics.losses.length - 1].toFixed(4),
        accuracy: metrics.accuracies[metrics.accuracies.length - 1].toFixed(4),
        fScore: metrics.fScores[metrics.fScores.length - 1].toFixed(4)
    });
    
    // Start monitoring
    console.log('Starting network monitoring...');
    await monitor.startMonitoring();
}

main().catch(console.error);