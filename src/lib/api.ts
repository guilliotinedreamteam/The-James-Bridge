export const api = {
  train: async () => {
    const response = await fetch("http://localhost:8000/train", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ config_path: "neurobridge.config.yaml", epochs: 10 }),
    });
    if (!response.ok) {
      throw new Error(`Train request failed: ${response.statusText}`);
    }
    return response.json();
  },
  evolve: async () => {
    const response = await fetch("http://localhost:8000/evolve", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ config_path: "neurobridge.config.yaml", generations: 100 }),
    });
    if (!response.ok) {
      throw new Error(`Evolve request failed: ${response.statusText}`);
    }
    return response.json();
  },
  synthesize: async (sequence: string) => {
    const response = await fetch("http://localhost:8000/synthesize", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ sequence }),
    });
    if (!response.ok) {
      throw new Error(`Synthesize request failed: ${response.statusText}`);
    }
    return response.json();
  },
};
