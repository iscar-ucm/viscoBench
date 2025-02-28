from src.simulator import SystemSimulator


if __name__ == "__main__":
    circuit_path = "cfg/circuit_elements.yaml"
    cfg_path = "cfg/data_gen_cfg.yaml"
    output_path = "data"

    generator = SystemSimulator()
    generator.load_values(circuit_path)
    generator.load_conditions(cfg_path)

    generator.generate_data(output_path)