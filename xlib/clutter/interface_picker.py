__authors__ = ['Kirk Scanlan, kirk.scanlan@utexas.edu']
__version__ = '1.0'
__history__ = {
    '1.0':
        {'date': 'January 21 2019',
         'author': 'Kirk Scanlan, UTIG',
         'info': 'interactive interface picker'}}

def picker(data, interfaces=[], color='viridis', snap_to='maximum', plt_final=False):
    '''
    algorithm for picking interface indices
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    from tkinter import messagebox

    if len(interfaces) == 0:
        kk = 0
        interfaces = np.full((np.size(data, axis=0), np.size(data, axis=1)), np.nan)
    else:
        kk = 1
    quit_picking = 'No'
    
    cmin = np.ceil(np.min(data) / 5) * 5
    cmax = np.floor(np.max(data) / 5) * 5

    while quit_picking == 'No':

        redefine_zoom = 'No'
        ll = 0

        # define the zoom bounds
        plt.figure()
        plt.imshow(np.transpose(data), aspect='auto', cmap=color)
        if kk != 0:
            plt.imshow(np.transpose(interfaces), aspect='auto')
        #fM = plt.get_current_fig_manager()
        #fM.window.showMaximized()
        plt.title('pick opposite corners of the zone of interest')
        plt.clim([cmin, cmax])
        plt.show()
        zoom = np.rint(plt.ginput(2, show_clicks=True, timeout=-1, mouse_pop=3)).astype(int)
        plt.close()

        # define the zoomed in area
        zoom_data = data[zoom[0, 0]:zoom[1, 0], zoom[0, 1]:zoom[1, 1]]

        while redefine_zoom == 'No':

            # pick the upper bound
            plt.figure()
            plt.imshow(np.transpose(zoom_data), aspect='auto', cmap=color)
            if kk != 0 or ll != 0:
                plt.imshow(np.transpose(interfaces[zoom[0, 0]:zoom[1, 0], zoom[0, 1]:zoom[1, 1]]),
                           aspect='auto')
            #fM = plt.get_current_fig_manager()
            #fM.window.showMaximized()
            plt.title('pick upper bounds of the reflection of interest - enter to end')
            plt.clim([cmin, cmax])
            plt.show()
            top = np.rint(plt.ginput(-1, show_clicks=True, timeout=-1, mouse_pop=3)).astype(int)
            plt.close()

            # pick the lower bound
            plt.figure()
            plt.imshow(np.transpose(zoom_data), aspect='auto', cmap=color)
            plt.plot(top[:, 0], top[:, 1], 'r')
            if kk != 0 or ll != 0:
                plt.imshow(np.transpose(interfaces[zoom[0, 0]:zoom[1, 0], zoom[0, 1]:zoom[1, 1]]),
                           aspect='auto')
            #fM = plt.get_current_fig_manager()
            #fM.window.showMaximized()
            plt.title('pick lower bounds of the reflection of interest - enter to end')
            plt.clim([cmin, cmax])
            plt.show()
            bottom = np.rint(plt.ginput(-1, show_clicks=True, timeout=-1, mouse_pop=3)).astype(int)
            plt.close()

            # define top and bottom bounds for each range line
            min_line = np.max((np.min(top[:, 0]), np.min(bottom[:, 0])))
            max_line = np.min((np.max(top[:, 0]), np.max(bottom[:, 0])))
            lines = np.arange(min_line, max_line + 1).astype(int)
            top_int = np.interp(lines, top[:, 0], top[:, 1]).astype(int)
            bottom_int = np.interp(lines, bottom[:, 0], bottom[:, 1]).astype(int)

            # snap to pick inside window
            if snap_to == 'maximum':
                picks = np.zeros((len(lines), 2), dtype=int)
            elif snap_to == 'all':
                picks = np.zeros((len(lines), 3), dtype=int)
            for ii in range(len(picks)):
                picks[ii, 0] = lines[ii]
                if snap_to == 'maximum':
                    temp = zoom_data[lines[ii], top_int[ii]:bottom_int[ii]]
                    picks[ii, 1] = np.argwhere(temp == np.max(temp))[0] + top_int[ii]
                elif snap_to == 'all':
                    picks[ii, 1] = top_int[ii]
                    picks[ii, 2] = bottom_int[ii]

            # append to interfaces array
            picks[:, 0] = picks[:, 0] + zoom[0, 0]
            picks[:, 1] = picks[:, 1] + zoom[0, 1]
            if snap_to == 'all':
                picks[:, 2] = picks[:, 2] + zoom[0, 1]
            ll += 1
            for ii in range(len(picks)):
                if snap_to == 'maximum':
                    interfaces[picks[ii, 0], picks[ii, 1]] = 1
                elif snap_to == 'all':
                    interfaces[picks[ii, 0], picks[ii, 1]:picks[ii, 2]] = np.ones((picks[ii, 2] - picks[ii, 1], ), dtype=int)
                
            # plot existing picks in this area
            plt.figure()
            plt.imshow(np.transpose(zoom_data), aspect='auto', cmap=color)
            plt.imshow(np.transpose(interfaces[zoom[0, 0]:zoom[1, 0], zoom[0, 1]:zoom[1, 1]]),
                       aspect='auto')
            #fM = plt.get_current_fig_manager()
            #fM.window.showMaximized()
            plt.title('picked interfaces in zoom area')
            plt.clim([cmin, cmax])
            plt.show()

            # query whether to re-define the zoom area
            if messagebox.askyesno("Python", "Quit picking in this zoom area?"):
                redefine_zoom = 'Yes'
            plt.close()

        kk += 1

        # query whether to continue picking
        if messagebox.askyesno("Python", "Quit picking?"):
            quit_picking = 'Yes'

    if plt_final:
        # plot the picked interfaces
        plt.figure()
        plt.imshow(np.transpose(data), aspect='auto', cmap=color)
        plt.imshow(np.transpose(interfaces))
        #fM = plt.get_current_fig_manager()
        #fM.window.showMaximized()
        plt.title('all picked interfaces')
        plt.clim([cmin, cmax])
        plt.show()

    return interfaces

def remover(data, interfaces, color='viridis', plt_final=False):
    '''
    algorithm for picking interface indices
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    from tkinter import messagebox

    kk = 1
    quit_picking = 'No'
    
    cmin = np.ceil(np.min(data) / 5) * 5
    cmax = np.floor(np.max(data) / 5) * 5

    while quit_picking == 'No':

        redefine_zoom = 'No'
        ll = 0

        # define the zoom bounds
        plt.figure()
        plt.imshow(np.transpose(data), aspect='auto', cmap=color)
        if kk != 0:
            plt.imshow(np.transpose(interfaces), aspect='auto')
        fM = plt.get_current_fig_manager()
        fM.window.showMaximized()
        plt.title('pick opposite corners of the zone of interest')
        plt.clim([cmin, cmax])
        plt.show()
        zoom = np.rint(plt.ginput(2, show_clicks=True, timeout=-1, mouse_pop=3)).astype(int)
        plt.close()

        # define the zoomed in area
        zoom_data = data[zoom[0, 0]:zoom[1, 0], zoom[0, 1]:zoom[1, 1]]
        zoom_interfaces = interfaces[zoom[0, 0]:zoom[1, 0], zoom[0, 1]:zoom[1, 1]]

        while redefine_zoom == 'No':

            # pick the upper bound
            plt.figure()
            plt.imshow(np.transpose(zoom_data), aspect='auto', cmap=color)
            if kk != 0 or ll != 0:
                plt.imshow(np.transpose(interfaces[zoom[0, 0]:zoom[1, 0], zoom[0, 1]:zoom[1, 1]]),
                           aspect='auto')
            fM = plt.get_current_fig_manager()
            fM.window.showMaximized()
            plt.title('pick upper bounds of the reflection to be removed - enter to end')
            plt.clim([cmin, cmax])
            top = np.rint(plt.ginput(-1, show_clicks=True, timeout=-1, mouse_pop=3)).astype(int)
            plt.close()

            # pick the lower bound
            plt.figure()
            plt.imshow(np.transpose(zoom_data), aspect='auto', cmap=color)
            plt.plot(top[:, 0], top[:, 1], 'r')
            if kk != 0 or ll != 0:
                plt.imshow(np.transpose(interfaces[zoom[0, 0]:zoom[1, 0], zoom[0, 1]:zoom[1, 1]]),
                           aspect='auto')
            fM = plt.get_current_fig_manager()
            fM.window.showMaximized()
            plt.title('pick lower bounds of the reflection to be removed - enter to end')
            plt.clim([cmin, cmax])
            bottom = np.rint(plt.ginput(-1, show_clicks=True, timeout=-1, mouse_pop=3)).astype(int)
            plt.close()

            # define top and bottom bounds for each range line
            min_line = np.max((np.min(top[:, 0]), np.min(bottom[:, 0])))
            max_line = np.min((np.max(top[:, 0]), np.max(bottom[:, 0])))
            lines = np.arange(min_line, max_line + 1).astype(int)
            top_int = np.interp(lines, top[:, 0], top[:, 1]).astype(int)
            bottom_int = np.interp(lines, bottom[:, 0], bottom[:, 1]).astype(int)

            # snap to pick inside window
            picks = np.zeros((len(lines), 2), dtype=int)
            for ii in range(len(picks)):
                temp = zoom_interfaces[lines[ii], top_int[ii]:bottom_int[ii]]
                picks[ii, 0] = lines[ii]
                if len(np.argwhere(temp == 1)) >= 1:
                    picks[ii, 1] = np.argwhere(temp == 1)[0] + top_int[ii]
                else:
                    picks[ii, 1] = top_int[ii]

            # remove pick from array
            picks[:, 0] = picks[:, 0] + zoom[0, 0]
            picks[:, 1] = picks[:, 1] + zoom[0, 1]
            ll += 1
            for ii in range(len(picks)):
                interfaces[picks[ii, 0], picks[ii, 1]] = np.nan
                
            # plot existing picks in this area
            plt.figure()
            plt.imshow(np.transpose(zoom_data), aspect='auto', cmap=color)
            plt.imshow(np.transpose(interfaces[zoom[0, 0]:zoom[1, 0], zoom[0, 1]:zoom[1, 1]]),
                       aspect='auto')
            fM = plt.get_current_fig_manager()
            fM.window.showMaximized()
            plt.title('picked interfaces in zoom area')
            plt.clim([cmin, cmax])

            # query whether to re-define the zoom area
            if messagebox.askyesno("Python", "Quit removing picks in this zoom area?"):
                redefine_zoom = 'Yes'
            plt.close()

        kk += 1

        # query whether to continue picking
        if messagebox.askyesno("Python", "Quit removing?"):
            quit_picking = 'Yes'

    if plt_final:
        # plot the picked interfaces
        plt.figure()
        plt.imshow(np.transpose(data), aspect='auto', cmap=color)
        plt.imshow(np.transpose(interfaces))
        fM = plt.get_current_fig_manager()
        fM.window.showMaximized()
        plt.title('all picked interfaces')
        plt.clim([cmin, cmax])

    return interfaces

def remover_mac(data, interfaces, color='viridis', plt_final=False):
    '''
    non-interactive algorithm for removing picked interfaces when working on
    mac
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    
    cmin = np.ceil(np.min(data) / 5) * 5
    cmax = np.floor(np.max(data) / 5) * 5

    # define the zoom bounds
    plt.figure()
    plt.imshow(np.transpose(data), aspect='auto', cmap=color)
    plt.imshow(np.transpose(interfaces), aspect='auto')
    fM = plt.get_current_fig_manager()
    fM.window.showMaximized()
    plt.title('pick opposite corners of the zone of interest')
    plt.clim([cmin, cmax])
    plt.show()
    zoom = np.rint(plt.ginput(2, show_clicks=True, timeout=-1, mouse_pop=3)).astype(int)
    plt.close()

    # define the zoomed in area
    zoom_data = data[zoom[0, 0]:zoom[1, 0], zoom[0, 1]:zoom[1, 1]]
    zoom_interfaces = interfaces[zoom[0, 0]:zoom[1, 0], zoom[0, 1]:zoom[1, 1]]


    # pick the upper bound
    plt.figure()
    plt.imshow(np.transpose(zoom_data), aspect='auto', cmap=color)
    plt.imshow(np.transpose(interfaces[zoom[0, 0]:zoom[1, 0], zoom[0, 1]:zoom[1, 1]]),
               aspect='auto')
    fM = plt.get_current_fig_manager()
    fM.window.showMaximized()
    plt.title('pick upper bounds of the reflection to be removed - enter to end')
    plt.clim([cmin, cmax])
    top = np.rint(plt.ginput(-1, show_clicks=True, timeout=-1, mouse_pop=3)).astype(int)
    plt.close()

    # pick the lower bound
    plt.figure()
    plt.imshow(np.transpose(zoom_data), aspect='auto', cmap=color)
    plt.plot(top[:, 0], top[:, 1], 'r')
    plt.imshow(np.transpose(interfaces[zoom[0, 0]:zoom[1, 0], zoom[0, 1]:zoom[1, 1]]),
                   aspect='auto')
    fM = plt.get_current_fig_manager()
    fM.window.showMaximized()
    plt.title('pick lower bounds of the reflection to be removed - enter to end')
    plt.clim([cmin, cmax])
    bottom = np.rint(plt.ginput(-1, show_clicks=True, timeout=-1, mouse_pop=3)).astype(int)
    plt.close()

    # define top and bottom bounds for each range line
    min_line = np.max((np.min(top[:, 0]), np.min(bottom[:, 0])))
    max_line = np.min((np.max(top[:, 0]), np.max(bottom[:, 0])))
    lines = np.arange(min_line, max_line + 1).astype(int)
    top_int = np.interp(lines, top[:, 0], top[:, 1]).astype(int)
    bottom_int = np.interp(lines, bottom[:, 0], bottom[:, 1]).astype(int)

    # snap to pick inside window
    picks = np.zeros((len(lines), 2), dtype=int)
    for ii in range(len(picks)):
        temp = zoom_interfaces[lines[ii], top_int[ii]:bottom_int[ii]]
        picks[ii, 0] = lines[ii]
        if len(np.argwhere(temp == 1)) >= 1:
            picks[ii, 1] = np.argwhere(temp == 1)[0] + top_int[ii]
        else:
            picks[ii, 1] = top_int[ii]

    # remove pick from array
    picks[:, 0] = picks[:, 0] + zoom[0, 0]
    picks[:, 1] = picks[:, 1] + zoom[0, 1]
    for ii in range(len(picks)):
        interfaces[picks[ii, 0], picks[ii, 1]] = np.nan

    if plt_final:
        # plot the picked interfaces
        plt.figure()
        plt.imshow(np.transpose(data), aspect='auto', cmap=color)
        plt.imshow(np.transpose(interfaces))
        fM = plt.get_current_fig_manager()
        fM.window.showMaximized()
        plt.title('all picked interfaces')
        plt.clim([cmin, cmax])

    return interfaces
