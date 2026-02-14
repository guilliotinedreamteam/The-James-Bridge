const API_BASE = "http://localhost:8000";

export const api = {
  status: async () => {
    const res = await fetch(`${API_BASE}/status`);
    if (!res.ok) throw new Error("Failed to fetch status");
    return res.json();
  },
  train: async (configPath: string = "neurobridge.config.yaml", epochs: number = 10) => {
    const res = await fetch(`${API_BASE}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config_path: configPath, epochs }),
    });
    if (!res.ok) throw new Error("Failed to start training");
    return res.json();
  },
  synthesize: async (sequence: string, outputPath?: string) => {
    const res = await fetch(`${API_BASE}/synthesize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence, output_path: outputPath }),
    });
    if (!res.ok) throw new Error("Failed to synthesize");
    return res.json();
  }
};
