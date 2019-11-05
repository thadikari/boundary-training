import adversarial


class StdStream:
    def __init__(self, old, prefix):
        self.old = old
        self.prefix = prefix
        self.lastchar = '\n'

    def write(self, string):
        self.old.write((self.prefix+'|' if self.lastchar=='\n' else '') + string)
        self.lastchar = string[-1]

    def flush(self):
        self.old.flush()


def main_mpi():
    from mpi4py import MPI
    import sys

    arrid = int(sys.argv[1]) if len(sys.argv)>1 else 0  # run array job on niagara
    comm = MPI.COMM_WORLD
    csiz = comm.Get_size()
    rank = comm.Get_rank()

    id = arrid*csiz + rank
    print('Array MPI job: arrid[%d], comm.size[%d], comm.rank[%d] --> job_id[%d]'%(arrid, csiz, rank, id))

    prefix = 'id%d'%id
    sys.stdout = StdStream(sys.stdout, prefix)
    sys.stderr = StdStream(sys.stderr, prefix)
    adversarial.main(id)


def main_seq():
    for id in range(0, 10):
        adversarial.main(id)

if __name__ == '__main__': main_seq() # main_seq/main_mpi
