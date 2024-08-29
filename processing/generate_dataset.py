from src.dataset_generator import DatasetGenerator


if __name__ == "__main__":
    circuit_path = "cfg/circuit_elements.yaml"
    cfg_path = "cfg/data_gen_cfg.yaml"
    output_path = "data"

    generator = DatasetGenerator()
    generator.load_values(circuit_path)
    generator.load_conditions(cfg_path)

    generator.generate_data(output_path)