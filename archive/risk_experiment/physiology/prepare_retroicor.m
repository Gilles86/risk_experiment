
function physio = prepare_retroicor(subject, session)
    subject
    session

    fieldstrength = str2num(session(1));
   
%     %% Create default parameter structure with all fields
     physio = tapas_physio_new();
% 
%     %% Individual Parameter settings. Modify to your need and remove default settings
     
         
     if session(3) == '1'
         runs = 1:4;
         task = 'mapper'
     elseif session(3) == '2'
         runs = 1:8;
         task = 'task'   
     else
        ME = MException('Not the right number of sessions');
        throw(ME)
     end
     
    if strcmp(subject, '08') && strcmp(session, '7t1')
        runs = [1 2 4 5]
    elseif strcmp(subject, '23') && strcmp(session, '7t1')
        runs = [1 2 3 5]
    end
    disp(runs)
    
    for run = runs
        close all;
        retroicor(subject, session, task, run)
    end
end