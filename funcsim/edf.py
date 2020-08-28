def makeedf(sample):
    # Return the empircal distiution function for a data sample.
    # "sample" should be some iterable object: e.g., a list,
    # an np.array, a pd.Series, or an 1-D xr.DataArray
    M = float(len(sample))

    def edf(v):
        return sum(map(lambda a: 1 if a <= v else 0, sample)) / M

    return edf