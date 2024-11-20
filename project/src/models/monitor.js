import * as tf from '@tensorflow/tfjs';

export class RealTimeMonitor {
    constructor(classifier) {
        this.classifier = classifier;
        this.detectionHistory = [];
        this.lastUpdate = Date.now();
    }

    async startMonitoring() {
        const monitorLoop = async () => {
            try {
                const data = await this.getNetworkData();
                const { predictions, warnings } = await this.classifier.predict(data);
                
                if (warnings.some(w => w)) {
                    console.log('⚠️ Potential intrusion detected!');
                    console.log('Statistics:', this.getStatistics());
                }
                
                this.updateHistory(predictions);
            } catch (error) {
                console.error('Monitoring error:', error);
            }
            
            setTimeout(monitorLoop, 1000);
        };
        
        await monitorLoop();
    }

    getStatistics() {
        return {
            totalDetections: this.detectionHistory.length,
            attackRatio: this.detectionHistory.filter(x => x > 0).length / this.detectionHistory.length,
            lastUpdate: new Date(this.lastUpdate).toISOString()
        };
    }

    async getNetworkData() {
        // Simulated network data collection
        return Array.from({ length: 80 }, () => Math.random());
    }

    updateHistory(predictions) {
        this.detectionHistory.push(...predictions);
        if (this.detectionHistory.length > 1000) {
            this.detectionHistory = this.detectionHistory.slice(-1000);
        }
    }
}