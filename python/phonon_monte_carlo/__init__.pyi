def run(input: str, sources: None | str, quiet: None | bool = False):
    r"""
    Run a Phonon Monte Carlo simulation with a config TOML file specified in `input`
    and a sources .parquet file specified in `sources`.

    Running from python
    ```
        run("/path/to/config.toml", "/path/to/sources.toml")
    ```
    """
