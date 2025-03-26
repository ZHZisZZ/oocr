def parse_pairings(pairings: list[str]) -> list[dict]:
    # Split each row into fields
    rows = [list(map(str.strip, row.split(","))) for row in pairings]
    # Extract header and rows
    keys = rows[0]
    dict_list = [dict(zip(keys, values)) for values in rows[1:]]
    return dict_list


if __name__ == "__main__":
    import omegaconf
    config = omegaconf.OmegaConf.load("oocr/two_hop/configs/city.yaml")
    dict_list = parse_pairings(config["pairings"])
    breakpoint()
