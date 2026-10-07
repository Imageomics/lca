# -*- coding: utf-8 -*-
import logging


logger = logging.getLogger("beta_stability")


def main():  # nocover
    # import utool

    # logger.info('Looks like the imports worked')
    # logger.info('utool = {!r}'.format(utool))
    # logger.info('utool.__file__ = {!r}'.format(utool.__file__))
    # logger.info('utool.__version__ = {!r}'.format(utool.__version__))

    import networkx

    logger.info('networkx = {!r}'.format(networkx))
    logger.info('networkx.__file__ = {!r}'.format(networkx.__file__))
    logger.info('networkx.__version__ = {!r}'.format(networkx.__version__))


if __name__ == '__main__':
    """
    CommandLine:
       python -m lca.__main__
    """
    main()
