import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import { Upload, X, FileText, Image, FileSpreadsheet } from "lucide-react";
import { cn } from "@/lib/utils";

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  file?: File;
}

interface FileUploadZoneProps {
  onFilesUploaded: (files: UploadedFile[]) => void;
}

export function FileUploadZone({ onFilesUploaded }: FileUploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const files = Array.from(e.dataTransfer.files);
      const uploadedFiles: UploadedFile[] = files.map((file) => ({
        id: Math.random().toString(36).substring(7),
        name: file.name,
        size: file.size,
        type: file.type,
        file: file,
      }));

      onFilesUploaded(uploadedFiles);
    },
    [onFilesUploaded]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files || []);
      const uploadedFiles: UploadedFile[] = files.map((file) => ({
        id: Math.random().toString(36).substring(7),
        name: file.name,
        size: file.size,
        type: file.type,
        file: file,
      }));

      onFilesUploaded(uploadedFiles);
    },
    [onFilesUploaded]
  );

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={cn(
        "border-2 border-dashed rounded-lg p-12 transition-all cursor-pointer relative overflow-hidden",
        isDragging
          ? "border-primary bg-primary/10 scale-[1.02]"
          : "border-border hover:border-primary/50 hover:bg-primary/5"
      )}
    >
      <input
        type="file"
        multiple
        onChange={handleFileSelect}
        className="hidden"
        id="file-upload"
        accept=".pdf,.doc,.docx,.xls,.xlsx,.png,.jpg,.jpeg,.txt"
      />
      <label htmlFor="file-upload" className="cursor-pointer">
        <div className="flex flex-col items-center text-center">
          <motion.div
            animate={isDragging ? { scale: 1.1, rotate: 5 } : { scale: 1, rotate: 0 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <Upload className="w-12 h-12 text-primary mb-4" />
          </motion.div>
          <p className="text-lg font-medium mb-2">
            Drop files here or click to browse
          </p>
          <p className="text-sm text-muted-foreground mb-4">
            Up to 50MB - Async processing for large documents
          </p>
          <div className="flex gap-2 flex-wrap justify-center">
            <span className="px-3 py-1 bg-red-500/20 text-red-400 rounded text-xs border border-red-500/30 hover:bg-red-500/30 transition-colors">
              PDF
            </span>
            <span className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded text-xs border border-blue-500/30 hover:bg-blue-500/30 transition-colors">
              Word
            </span>
            <span className="px-3 py-1 bg-green-500/20 text-green-400 rounded text-xs border border-green-500/30 hover:bg-green-500/30 transition-colors">
              Excel
            </span>
            <span className="px-3 py-1 bg-purple-500/20 text-purple-400 rounded text-xs border border-purple-500/30 hover:bg-purple-500/30 transition-colors">
              PNG
            </span>
            <span className="px-3 py-1 bg-orange-500/20 text-orange-400 rounded text-xs border border-orange-500/30 hover:bg-orange-500/30 transition-colors">
              JPG
            </span>
            <span className="px-3 py-1 bg-slate-500/20 text-slate-400 rounded text-xs border border-slate-500/30 hover:bg-slate-500/30 transition-colors">
              Text
            </span>
          </div>
        </div>
      </label>
    </div>
  );
}

interface FileListProps {
  files: UploadedFile[];
  onRemove: (id: string) => void;
}

export function FileList({ files, onRemove }: FileListProps) {
  const getFileIcon = (type: string) => {
    if (type.includes("pdf")) return <FileText className="w-4 h-4 text-red-400" />;
    if (type.includes("image")) return <Image className="w-4 h-4 text-purple-400" />;
    if (type.includes("spreadsheet") || type.includes("excel"))
      return <FileSpreadsheet className="w-4 h-4 text-green-400" />;
    return <FileText className="w-4 h-4 text-blue-400" />;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  if (files.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground text-sm">
        Upload documents to see sources here
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {files.map((file) => (
        <motion.div
          key={file.id}
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, x: -10 }}
          whileHover={{ scale: 1.02 }}
          className="flex items-center justify-between p-3 rounded-lg bg-secondary/50 border border-border hover:border-primary/30 transition-all glass-effect"
        >
          <div className="flex items-center gap-3 flex-1 min-w-0">
            {getFileIcon(file.type)}
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate">{file.name}</p>
              <p className="text-xs text-muted-foreground">
                {formatFileSize(file.size)}
              </p>
            </div>
          </div>
          <button
            onClick={() => onRemove(file.id)}
            className="p-1 hover:bg-destructive/10 rounded transition-colors group"
          >
            <X className="w-4 h-4 text-muted-foreground group-hover:text-red-400 transition-colors" />
          </button>
        </motion.div>
      ))}
    </div>
  );
}
