import * as tf from '@tensorflow/tfjs';

export class MetricsTracker {
    constructor() {
        this.losses = [];
        this.accuracies = [];
        this.fScores = [];
        this.epochs = [];
    }

    updateMetrics(epoch, loss, predictions, labels) {
        // Calculate accuracy
        const accuracy = this.calculateAccuracy(predictions, labels);
        
        // Calculate F-score
        const fScore = this.calculateFScore(predictions, labels);
        
        // Store metrics
        this.epochs.push(epoch);
        this.losses.push(loss);
        this.accuracies.push(accuracy);
        this.fScores.push(fScore);
    }

    calculateAccuracy(predictions, labels) {
        const correctPredictions = predictions.map((pred, i) => 
            tf.argMax(pred).dataSync()[0] === tf.argMax(labels[i]).dataSync()[0]
        );
        return correctPredictions.reduce((a, b) => a + b, 0) / predictions.length;
    }

    calculateFScore(predictions, labels) {
        const tp = predictions.reduce((sum, pred, i) => {
            const predClass = tf.argMax(pred).dataSync()[0];
            const trueClass = tf.argMax(labels[i]).dataSync()[0];
            return sum + (predClass === trueClass && predClass === 1 ? 1 : 0);
        }, 0);

        const fp = predictions.reduce((sum, pred, i) => {
            const predClass = tf.argMax(pred).dataSync()[0];
            const trueClass = tf.argMax(labels[i]).dataSync()[0];
            return sum + (predClass === 1 && trueClass === 0 ? 1 : 0);
        }, 0);

        const fn = predictions.reduce((sum, pred, i) => {
            const predClass = tf.argMax(pred).dataSync()[0];
            const trueClass = tf.argMax(labels[i]).dataSync()[0];
            return sum + (predClass === 0 && trueClass === 1 ? 1 : 0);
        }, 0);

        const precision = tp / (tp + fp) || 0;
        const recall = tp / (tp + fn) || 0;
        
        return 2 * (precision * recall) / (precision + recall) || 0;
    }

    plotMetrics() {
        const plotData = [
            {
                x: this.epochs,
                y: this.losses,
                type: 'scatter',
                name: 'Loss Rate',
                line: { color: 'red' }
            },
            {
                x: this.epochs,
                y: this.accuracies,
                type: 'scatter',
                name: 'Recognition Rate',
                line: { color: 'blue' }
            },
            {
                x: this.epochs,
                y: this.fScores,
                type: 'scatter',
                name: 'F-Score',
                line: { color: 'green' }
            }
        ];

        const layout = {
            title: 'Training Metrics Over Time',
            xaxis: { title: 'Epoch' },
            yaxis: { title: 'Value' },
            showlegend: true
        };

        return { data: plotData, layout };
    }
}