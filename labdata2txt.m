function labdata2txt(delimiter)
%LABDATA2TXT Extracts useful information from lab .mat files
%   Extracts data points for analysis in programs other than MATLAB

    % Find all directories starting with "Lab-"
    lab_dirs = ls('Lab-*');
    n_dirs = size(lab_dirs);
    n_dirs = n_dirs(1);
    delimiter = strcat('%d', delimiter);
    
    for index_i = 1:1:n_dirs
        % Find all ".mat" files in directory beginning with "Lab-"
        dir_name = lab_dirs(index_i,:);
        mat_files = ls(strcat(dir_name, '/*.mat'));
        n_files = size(mat_files);
        n_files = n_files(1);
        
        for index_j = 1:1:n_files
            % Extract useful information from ".mat" file and write to
            % output file with commas separating values, extention ".txt"
            input_filename = strcat(dir_name, '/', mat_files(index_j, :));
            output_filename = strrep(input_filename, '.mat', '.txt');
            data_struct = load(input_filename);
            data_cell = struct2cell(data_struct);
            
            data_dimensions = size(data_cell{1, 1}.X.Data);
            var_dimensions = size(data_cell{1,1}.Y);
            data_matrix = zeros(var_dimensions(2) + 1, data_dimensions(2));
            data_matrix(1, :) = data_cell{1, 1}.X.Data;
             
            for index_k = 1:1:var_dimensions(2)
                data_matrix(index_k+1, :) = data_cell{1, 1}.Y(index_k).Data;
            end
            
            output_file = fopen(output_filename, 'wt');
            for index_l = 1:1:var_dimensions(2)+1
                fprintf(output_file, delimiter, data_matrix(index_l, :));
                fprintf(output_file, '\n');
            end
            fclose(output_file);
        end
    end
end

