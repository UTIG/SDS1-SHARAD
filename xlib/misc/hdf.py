class hdf:
    def __init__(self, path, **kwargs):
        import pandas as pd
        self.file = pd.HDFStore(path, **kwargs)

    def keys(self, group, full_path=True):
        node = self.file.get_node(group)
        if (node is None):
            print('ERROR: No data for '+group+' in input file.')
            return None
        else:
            if (full_path):
                return [group+'/'+key for key in node._v_children.keys()]
            else:
                return [key for key in node._v_children.keys()]

    def to_dict(self, group):
        """
        Return one group of the h5file as a dictionary
        """
        keys = self.keys(group, full_path=False)
        if (keys is not None):
            return {key: self.file[group+'/'+key] for key in keys}
        else:
            return None

    def to_DataFrame(self, group):
        """
        Return the group of the h5file as a data frame
        """
        return self.file.get(group)

    def to_nparray(self, group,
                   col_names=['spot_lat', 'spot_lon', 'spot_radius']):
        """
        If col_names is None all columns are returned.
        """
        import numpy as np
        data = self.to_dict(group)
        if (col_names is not None):
            return np.concatenate([data[key][col_names].values
                                   for key in data.keys()])
        else:
            return np.concatenate([data[key].values for key in data.keys()])

    def close(self):
        self.file.close()

    def save_dict(self, group, grouped_data, verbose=False):
        keys = grouped_data.keys()
        if (verbose):
            from pydlr.misc.prog import Prog
            pr = Prog(keys)
        for key in keys:
            self.file[group+'/'+key] = grouped_data[key]
            if verbose:
                pr.print_Prog(key, appendix=' | '+key)
        if verbose:
            pr.close_Prog()
            print('Data saved to '+self.file._path)

    def get_label(self, formatted=True):
        """
        Gets the label information from a HDF file.
        There are some special label information entries:
            1. "date" - date of creation of the file (always included)
            2. "user" - user who created the file (always included)
            3. "info" - description of the data (should be included)
        """
        try:
            label = self.file['label']
        except:
            label = None
        if (label is not None):
            if formatted:
                print_str = 'Using input file: ' + self.file._path + '\n'
                print_str += ('Description: ' +
                              label.get('info', 'not provided') + ' \n')
                if label.get('source') is not None:
                    print_str += 'Source: ' + label.get('source', '') + ' \n'
                print_str += 'Created ' + label['date'] + ' '
                print_str += 'by ' + label['user'] + '.\n'
                for key in label.keys():
                    if not(key in ['info', 'user', 'date', 'source']):
                        print_str += key+': '+str(label[key]) + '\n'
                return print_str[:-1]
            else:
                return dict(label)
        else:
            return 'No label information available in ' + self.file._path + '.'

    def update_label(self, **kwargs):
        """
        Example
        -------
        >>> h5 = hdf(path)
        >>> h5.update_label(info='Hello', other_keyword='value')
        >>> h5.close()
        """
        label = self.get_label(formatted=False)
        label.update(kwargs)
        self.save_label(**label)

    def save_label(self, **kwargs):
        import pandas as pd
        import time
        import os
        import warnings
        kwargs.update({'date': time.strftime("%d.%m.%Y-%H:%M:%S",
                                             time.localtime()),
                       'user': os.getenv('USER')})
        # disable the PerformanceWarning from the HDF:
        #     Your performance may suffer as PyTables will pickle object types
        #     that it cannot map directly to c-types.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.file['label'] = pd.Series(kwargs)
