import logging 

class Cands_handler:
    '''
    To be implemented
    '''
    def __init__(self, outname):
        self.outname = outname
        self.open_outfile()

    def open_outfile(self):
        self.f = open(self.outname, 'w')
        self.f.write("SNR\tboxcar\tDM\tsamp\tngroup\n")
    
    def write_cands(self, cands):
        if len(cands) > 0:
            logging.debug(f"Writing {len(cands)} cands to file")
            for cand in cands:
                for field in cand:
                    self.f.write(str(field) + "\t")
                self.f.write("\n")
        else:
            logging.debug("No cands to write")
        
