const API_BASE = "http://localhost:8000";

export const api = {
    async train(configPath: string = "neurobridge.config.yaml", epochs: number = 10) {
        const response = await fetch(`${API_BASE}/train`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ config_path: configPath, epochs }),
        });
        if (!response.ok) {
            throw new Error(`Failed to start training: ${response.statusText}`);
        }
        return response.json();
    },

    async evolve(generations: number = 100) {
        const response = await fetch(`${API_BASE}/evolve`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ generations }),
        });
        if (!response.ok) {
            throw new Error(`Failed to start evolution: ${response.statusText}`);
        }
        return response.json();
    },

    async synthesize(sequence: string, outputPath?: string) {
        const body: any = { sequence };
        if (outputPath) body.output_path = outputPath;

        const response = await fetch(`${API_BASE}/synthesize`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(body),
        });
        if (!response.ok) {
            throw new Error(`Failed to synthesize speech: ${response.statusText}`);
        }
        return response.json();
    },

    async getStatus() {
        const response = await fetch(`${API_BASE}/status`);
        if (!response.ok) {
             throw new Error(`Failed to get status: ${response.statusText}`);
        }
        return response.json();
    }
};
