import gzip

def read_gzip(file_path: str):
    """
    Opens and reads a gzip file as ascii stream. 
    Args:
        file_path (str): path to gzip file. 

    Returns:
        Stream of lines decoded as ascii from the gzip file
    """
    with gzip.open(file_path, "rb") as file:
        for line in file:
            yield line.decode('ascii')

def parse_contigs(file_path: str):
    """
    Opens and reads a gzipped assembly/contigs file. 
    Args:
        file_path (str): path to gzip contigs file. 

    Returns:
        Stream of contig sequences.
    """
    stream = read_gzip(file_path)

    for line in stream:
        if line[0] == ">":
            pass
        else:
            yield line

def kmerizer(seq: str, kmer: int = 6):
    """
    Converts a sequence of nucleotides into a list of kmers 
    Args:
        file_path (str): path to gzip contigs file. 

    Returns:
        list: list of kmers of defined size.
    """
    idx = 0
    tmp = []
    lst = []
    for character in seq:
        tmp.append(character)
        idx += 1
        if len(tmp) >= kmer:
            lst.append("".join(tmp))
            tmp = []
    return lst