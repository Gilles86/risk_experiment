function prepare_retroicor_all(subject)
    for session = ["3t1", '3t2', '7t1', '7t2']        
        session
        prepare_retroicor(subject, char(session));
        close all;
    end
    
end