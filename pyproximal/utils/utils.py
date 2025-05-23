from typing import Optional, List, Union, Any # For type hints
import types # For ModuleType

# scooby is a soft dependency for pyprox
try:
    from scooby import Report as ScoobyReport # type: ignore
except ImportError:
    # Define a placeholder ScoobyReport if scooby is not installed
    class ScoobyReport: # type: ignore
        def __init__(self, additional: Optional[Union[types.ModuleType, List[types.ModuleType]]],
                     core: List[str],
                     optional: List[str],
                     ncol: int,
                     text_width: int,
                     sort: bool) -> None:
            print("\nNOTE: `pyprox.Report` requires `scooby`. Install it via"
                  "\n      `pip install scooby` or "
                  "`conda install -c conda-forge scooby`.\n")


class Report(ScoobyReport):
    r"""Print date, time, and version information.

    Use ``scooby`` to print date, time, and package version information in any
    environment (Jupyter notebook, IPython console, Python console, QT
    console), either as html-table (notebook) or as plain text (anywhere).

    Always shown are the OS, number of CPU(s), ``numpy``, ``scipy``,
    ``pylops``, ``pyprox``, ``sys.version``, and time/date.

    Additionally shown are, if they can be imported, ``IPython``, ``numba``,
    and ``matplotlib``. It also shows MKL information, if available.

    All modules provided in ``add_pckg`` are also shown.

    .. note::

        The package ``scooby`` has to be installed in order to use ``Report``:
        ``pip install scooby`` or ``conda install -c conda-forge scooby``.


    Parameters
    ----------
    add_pckg : packages, optional
        Package or list of packages to add to output information (must be
        imported beforehand).

    ncol : int, optional
        Number of package-columns in html table (no effect in text-version);
        Defaults to 3.

    text_width : int, optional
        The text width for non-HTML display modes

    sort : bool, optional
        Sort the packages when the report is shown


    Examples
    --------
    >>> import pytest
    >>> import dateutil
    >>> from pyproximal import Report
    >>> Report()                            # Default values
    >>> Report(pytest)                      # Provide additional package
    >>> Report([pytest, dateutil], ncol=5)  # Set nr of columns

    """

    def __init__(self, add_pckg: Optional[Union[types.ModuleType, List[types.ModuleType], str, List[str]]] = None,
                 ncol: int = 3, text_width: int = 80, sort: bool = False) -> None:
        """Initiate a scooby.Report instance."""

        # Mandatory packages.
        core: List[str] = ['numpy', 'scipy', 'pylops', 'pyprox']

        # Optional packages.
        optional: List[str] = ['IPython', 'matplotlib', 'numba']
        
        # ScoobyReport's 'additional' parameter can take a module, a list of modules,
        # a package name string, or a list of package name strings.
        # Our add_pckg should align with this.
        processed_add_pckg: Any # Let Scooby handle the union of types for 'additional'
        if add_pckg is None:
            processed_add_pckg = None
        elif isinstance(add_pckg, (types.ModuleType, str)) or \
             (isinstance(add_pckg, list) and all(isinstance(item, (types.ModuleType, str)) for item in add_pckg)):
            processed_add_pckg = add_pckg
        else:
            # If it's some other type, Scooby might handle it or raise an error.
            # For stricter typing, one might raise a TypeError here for unsupported add_pckg types.
            # However, Scooby's own typing for 'additional' is quite broad (Any).
            # For now, pass it through.
            processed_add_pckg = add_pckg


        super().__init__(additional=processed_add_pckg, core=core, optional=optional,
                         ncol=ncol, text_width=text_width, sort=sort)
